/*
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include "config.h"

#include <windows.h>

#define COBJMACROS

#include <initguid.h>
#include <d3d11.h>
#if CONFIG_VULKAN || CONFIG_CUDA
#include <d3dcompiler.h>
#endif
#include <dxgi1_2.h>

#if HAVE_DXGIDEBUG_H
#include <dxgidebug.h>
#endif

#include "avassert.h"
#include "common.h"
#include "hwcontext.h"
#include "hwcontext_d3d11va.h"
#include "hwcontext_d3d11va_internal.h"
#include "hwcontext_internal.h"
#if CONFIG_CUDA
#include <d3d11_4.h>
#include "cuda_check.h"
#include "hwcontext_cuda_internal.h"
#define CHECK_CU(x) FF_CUDA_CHECK_DL(cuda_cu, cu, x)
#endif
#if CONFIG_VULKAN
#include "hwcontext_vulkan.h"
#endif
#include "imgutils.h"
#include "mem.h"
#include "pixdesc.h"
#include "pixfmt.h"
#include "thread.h"
#include "compat/w32dlfcn.h"

#define MAX_ARRAY_SIZE 64 // Driver specification limits ArraySize to 64 for decoder-bound resources

typedef HRESULT(WINAPI *PFN_CREATE_DXGI_FACTORY)(REFIID riid, void **ppFactory);

static AVOnce functions_loaded = AV_ONCE_INIT;

static PFN_CREATE_DXGI_FACTORY mCreateDXGIFactory;
static PFN_D3D11_CREATE_DEVICE mD3D11CreateDevice;

static av_cold void load_functions(void)
{
#if !HAVE_UWP
    // We let these "leak" - this is fine, as unloading has no great benefit, and
    // Windows will mark a DLL as loaded forever if its internal refcount overflows
    // from too many LoadLibrary calls.
    HANDLE d3dlib, dxgilib;

    d3dlib  = dlopen("d3d11.dll", 0);
    dxgilib = dlopen("dxgi.dll", 0);
    if (!d3dlib || !dxgilib)
        return;

    mD3D11CreateDevice = (PFN_D3D11_CREATE_DEVICE) GetProcAddress(d3dlib, "D3D11CreateDevice");
    mCreateDXGIFactory = (PFN_CREATE_DXGI_FACTORY) GetProcAddress(dxgilib, "CreateDXGIFactory1");
    if (!mCreateDXGIFactory)
        mCreateDXGIFactory = (PFN_CREATE_DXGI_FACTORY) GetProcAddress(dxgilib, "CreateDXGIFactory");
#else
    // In UWP (which lacks LoadLibrary), CreateDXGIFactory isn't available,
    // only CreateDXGIFactory1
    mD3D11CreateDevice = (PFN_D3D11_CREATE_DEVICE) D3D11CreateDevice;
    mCreateDXGIFactory = (PFN_CREATE_DXGI_FACTORY) CreateDXGIFactory1;
#endif
}

typedef struct D3D11VAFramesContext {
    /**
     * The public AVD3D11VAFramesContext. See hwcontext_d3d11va.h for it.
     */
    AVD3D11VAFramesContext p;

    int nb_surfaces;
    int nb_surfaces_used;

    DXGI_FORMAT format;

    ID3D11Texture2D *staging_texture;

#if CONFIG_CUDA
    AVMutex                  cuda_lock;
    int                      cuda_lock_init;
    struct D3D11CudaInterop *cuda_interop;
#endif
} D3D11VAFramesContext;

#if CONFIG_CUDA
static int d3d11va_cuda_luid(AVCUDADeviceContext *cu_hw, LUID *luid);
static void d3d11va_cuda_interops_free(AVHWFramesContext *ctx);
static int d3d11va_cuda_transfer_data(AVHWFramesContext *ctx, AVFrame *dst,
                                      const AVFrame *src);
#endif

static const struct {
    DXGI_FORMAT d3d_format;
    enum AVPixelFormat pix_fmt;
} supported_formats[] = {
    { DXGI_FORMAT_NV12,         AV_PIX_FMT_NV12 },
    { DXGI_FORMAT_P010,         AV_PIX_FMT_P010 },
    { DXGI_FORMAT_B8G8R8A8_UNORM,    AV_PIX_FMT_BGRA },
    { DXGI_FORMAT_R10G10B10A2_UNORM, AV_PIX_FMT_X2BGR10 },
    { DXGI_FORMAT_R16G16B16A16_FLOAT, AV_PIX_FMT_RGBAF16 },
    { DXGI_FORMAT_AYUV,         AV_PIX_FMT_VUYX },
    { DXGI_FORMAT_YUY2,         AV_PIX_FMT_YUYV422 },
    { DXGI_FORMAT_Y210,         AV_PIX_FMT_Y210 },
    { DXGI_FORMAT_Y410,         AV_PIX_FMT_XV30 },
    { DXGI_FORMAT_P016,         AV_PIX_FMT_P016 },
    { DXGI_FORMAT_Y216,         AV_PIX_FMT_Y216 },
    { DXGI_FORMAT_Y416,         AV_PIX_FMT_XV48 },
    // There is no 12bit pixel format defined in DXGI_FORMAT*, use 16bit to compatible
    // with 12 bit AV_PIX_FMT* formats.
    { DXGI_FORMAT_P016,         AV_PIX_FMT_P012 },
    { DXGI_FORMAT_Y216,         AV_PIX_FMT_Y212 },
    { DXGI_FORMAT_Y416,         AV_PIX_FMT_XV36 },
    // Special opaque formats. The pix_fmt is merely a place holder, as the
    // opaque format cannot be accessed directly.
    { DXGI_FORMAT_420_OPAQUE,   AV_PIX_FMT_YUV420P },
};

static void d3d11va_default_lock(void *ctx)
{
    WaitForSingleObjectEx(ctx, INFINITE, FALSE);
}

static void d3d11va_default_unlock(void *ctx)
{
    ReleaseMutex(ctx);
}

static void d3d11va_frames_uninit(AVHWFramesContext *ctx)
{
    D3D11VAFramesContext *s = ctx->hwctx;
    AVD3D11VAFramesContext *frames_hwctx = &s->p;

#if CONFIG_CUDA
    d3d11va_cuda_interops_free(ctx);
    if (s->cuda_lock_init) {
        ff_mutex_destroy(&s->cuda_lock);
        s->cuda_lock_init = 0;
    }
#endif

    if (frames_hwctx->texture)
        ID3D11Texture2D_Release(frames_hwctx->texture);
    frames_hwctx->texture = NULL;

    if (s->staging_texture)
        ID3D11Texture2D_Release(s->staging_texture);
    s->staging_texture = NULL;

    av_freep(&frames_hwctx->texture_infos);
}

static int d3d11va_frames_get_constraints(AVHWDeviceContext *ctx,
                                          const void *hwconfig,
                                          AVHWFramesConstraints *constraints)
{
    AVD3D11VADeviceContext *device_hwctx = ctx->hwctx;
    int nb_sw_formats = 0;
    HRESULT hr;
    int i;

    constraints->valid_sw_formats = av_malloc_array(FF_ARRAY_ELEMS(supported_formats) + 1,
                                                    sizeof(*constraints->valid_sw_formats));
    if (!constraints->valid_sw_formats)
        return AVERROR(ENOMEM);

    for (i = 0; i < FF_ARRAY_ELEMS(supported_formats); i++) {
        UINT format_support = 0;
        hr = ID3D11Device_CheckFormatSupport(device_hwctx->device, supported_formats[i].d3d_format, &format_support);
        if (SUCCEEDED(hr) && (format_support & D3D11_FORMAT_SUPPORT_TEXTURE2D))
            constraints->valid_sw_formats[nb_sw_formats++] = supported_formats[i].pix_fmt;
    }
    constraints->valid_sw_formats[nb_sw_formats] = AV_PIX_FMT_NONE;

    constraints->valid_hw_formats = av_malloc_array(2, sizeof(*constraints->valid_hw_formats));
    if (!constraints->valid_hw_formats)
        return AVERROR(ENOMEM);

    constraints->valid_hw_formats[0] = AV_PIX_FMT_D3D11;
    constraints->valid_hw_formats[1] = AV_PIX_FMT_NONE;

    return 0;
}

static void free_texture(void *opaque, uint8_t *data)
{
    ID3D11Texture2D_Release((ID3D11Texture2D *)opaque);
    av_free(data);
}

static AVBufferRef *wrap_texture_buf(AVHWFramesContext *ctx, ID3D11Texture2D *tex, int index)
{
    AVBufferRef *buf;
    AVD3D11FrameDescriptor         *desc = av_mallocz(sizeof(*desc));
    D3D11VAFramesContext              *s = ctx->hwctx;
    AVD3D11VAFramesContext *frames_hwctx = &s->p;
    if (!desc) {
        ID3D11Texture2D_Release(tex);
        return NULL;
    }

    if (s->nb_surfaces <= s->nb_surfaces_used) {
        frames_hwctx->texture_infos = av_realloc_f(frames_hwctx->texture_infos,
                                                   s->nb_surfaces_used + 1,
                                                   sizeof(*frames_hwctx->texture_infos));
        if (!frames_hwctx->texture_infos) {
            ID3D11Texture2D_Release(tex);
            av_free(desc);
            return NULL;
        }
        s->nb_surfaces = s->nb_surfaces_used + 1;
    }

    frames_hwctx->texture_infos[s->nb_surfaces_used].texture = tex;
    frames_hwctx->texture_infos[s->nb_surfaces_used].index = index;
    s->nb_surfaces_used++;

    desc->texture = tex;
    desc->index   = index;

    buf = av_buffer_create((uint8_t *)desc, sizeof(*desc), free_texture, tex, 0);
    if (!buf) {
        ID3D11Texture2D_Release(tex);
        av_free(desc);
        return NULL;
    }

    return buf;
}

static AVBufferRef *d3d11va_alloc_single(AVHWFramesContext *ctx)
{
    D3D11VAFramesContext       *s = ctx->hwctx;
    AVD3D11VAFramesContext *hwctx = &s->p;
    AVD3D11VADeviceContext *device_hwctx = ctx->device_ctx->hwctx;
    HRESULT hr;
    ID3D11Texture2D *tex;
    D3D11_TEXTURE2D_DESC texDesc = {
        .Width      = ctx->width,
        .Height     = ctx->height,
        .MipLevels  = 1,
        .Format     = s->format,
        .SampleDesc = { .Count = 1 },
        .ArraySize  = 1,
        .Usage      = D3D11_USAGE_DEFAULT,
        .BindFlags  = hwctx->BindFlags,
        .MiscFlags  = hwctx->MiscFlags,
    };

    hr = ID3D11Device_CreateTexture2D(device_hwctx->device, &texDesc, NULL, &tex);
    if (FAILED(hr)) {
        av_log(ctx, AV_LOG_ERROR, "Could not create the texture (%lx)\n", (long)hr);
        return NULL;
    }

    return wrap_texture_buf(ctx, tex, 0);
}

static AVBufferRef *d3d11va_pool_alloc(void *opaque, size_t size)
{
    AVHWFramesContext        *ctx = (AVHWFramesContext*)opaque;
    D3D11VAFramesContext       *s = ctx->hwctx;
    AVD3D11VAFramesContext *hwctx = &s->p;
    D3D11_TEXTURE2D_DESC  texDesc;

    if (!hwctx->texture)
        return d3d11va_alloc_single(ctx);

    ID3D11Texture2D_GetDesc(hwctx->texture, &texDesc);

    if (s->nb_surfaces_used >= texDesc.ArraySize) {
        av_log(ctx, AV_LOG_ERROR, "Static surface pool size exceeded.\n");
        return NULL;
    }

    ID3D11Texture2D_AddRef(hwctx->texture);
    return wrap_texture_buf(ctx, hwctx->texture, s->nb_surfaces_used);
}

static int d3d11va_frames_init(AVHWFramesContext *ctx)
{
    AVD3D11VADeviceContext *device_hwctx = ctx->device_ctx->hwctx;
    D3D11VAFramesContext              *s = ctx->hwctx;
    AVD3D11VAFramesContext        *hwctx = &s->p;

    int i;
    HRESULT hr;
    D3D11_TEXTURE2D_DESC texDesc;

#if CONFIG_CUDA
    if (ff_mutex_init(&s->cuda_lock, NULL))
        return AVERROR(ENOMEM);
    s->cuda_lock_init = 1;
#endif

    for (i = 0; i < FF_ARRAY_ELEMS(supported_formats); i++) {
        if (ctx->sw_format == supported_formats[i].pix_fmt) {
            s->format = supported_formats[i].d3d_format;
            break;
        }
    }
    if (i == FF_ARRAY_ELEMS(supported_formats)) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported pixel format: %s\n",
               av_get_pix_fmt_name(ctx->sw_format));
        return AVERROR(EINVAL);
    }

    hwctx->BindFlags |= device_hwctx->BindFlags;
    hwctx->MiscFlags |= device_hwctx->MiscFlags;

    ctx->initial_pool_size = FFMIN(ctx->initial_pool_size, MAX_ARRAY_SIZE);

    texDesc = (D3D11_TEXTURE2D_DESC){
        .Width      = ctx->width,
        .Height     = ctx->height,
        .MipLevels  = 1,
        .Format     = s->format,
        .SampleDesc = { .Count = 1 },
        .ArraySize  = ctx->initial_pool_size,
        .Usage      = D3D11_USAGE_DEFAULT,
        .BindFlags  = hwctx->BindFlags,
        .MiscFlags  = hwctx->MiscFlags,
    };

    if (hwctx->texture) {
        D3D11_TEXTURE2D_DESC texDesc2;
        ID3D11Texture2D_GetDesc(hwctx->texture, &texDesc2);

        if (texDesc.Width != texDesc2.Width ||
            texDesc.Height != texDesc2.Height ||
            texDesc.Format != texDesc2.Format) {
            av_log(ctx, AV_LOG_ERROR, "User-provided texture has mismatching parameters\n");
            return AVERROR(EINVAL);
        }

        ctx->initial_pool_size = texDesc2.ArraySize;
        hwctx->BindFlags = texDesc2.BindFlags;
        hwctx->MiscFlags = texDesc2.MiscFlags;
    } else if (texDesc.ArraySize > 0) {
        hr = ID3D11Device_CreateTexture2D(device_hwctx->device, &texDesc, NULL, &hwctx->texture);
        if (FAILED(hr)) {
            av_log(ctx, AV_LOG_ERROR, "Could not create the texture (%lx)\n", (long)hr);
            return AVERROR_UNKNOWN;
        }
    }

    hwctx->texture_infos = av_realloc_f(NULL, ctx->initial_pool_size, sizeof(*hwctx->texture_infos));
    if (!hwctx->texture_infos)
        return AVERROR(ENOMEM);
    s->nb_surfaces = ctx->initial_pool_size;

    ffhwframesctx(ctx)->pool_internal =
        av_buffer_pool_init2(sizeof(AVD3D11FrameDescriptor),
                             ctx, d3d11va_pool_alloc, NULL);
    if (!ffhwframesctx(ctx)->pool_internal)
        return AVERROR(ENOMEM);

    return 0;
}

static int d3d11va_get_buffer(AVHWFramesContext *ctx, AVFrame *frame)
{
    AVD3D11FrameDescriptor *desc;

    frame->buf[0] = av_buffer_pool_get(ctx->pool);
    if (!frame->buf[0])
        return AVERROR(ENOMEM);

    desc = (AVD3D11FrameDescriptor *)frame->buf[0]->data;

    frame->data[0] = (uint8_t *)desc->texture;
    frame->data[1] = (uint8_t *)desc->index;
    frame->format  = AV_PIX_FMT_D3D11;
    frame->width   = ctx->width;
    frame->height  = ctx->height;

    return 0;
}

static int d3d11va_transfer_get_formats(AVHWFramesContext *ctx,
                                        enum AVHWFrameTransferDirection dir,
                                        enum AVPixelFormat **formats)
{
    D3D11VAFramesContext *s = ctx->hwctx;
    enum AVPixelFormat *fmts;
    int n = 0;

    fmts = av_malloc_array(4, sizeof(*fmts));
    if (!fmts)
        return AVERROR(ENOMEM);

    // Don't signal support for opaque formats. Actual access would fail.
    if (s->format != DXGI_FORMAT_420_OPAQUE) {
        fmts[n++] = ctx->sw_format;
#if CONFIG_VULKAN
        fmts[n++] = AV_PIX_FMT_VULKAN;
#endif
#if CONFIG_CUDA
        fmts[n++] = AV_PIX_FMT_CUDA;
#endif
    }
    fmts[n] = AV_PIX_FMT_NONE;

    *formats = fmts;

    return 0;
}

static int d3d11va_create_staging_texture(AVHWFramesContext *ctx, DXGI_FORMAT format)
{
    AVD3D11VADeviceContext *device_hwctx = ctx->device_ctx->hwctx;
    D3D11VAFramesContext              *s = ctx->hwctx;
    HRESULT hr;
    D3D11_TEXTURE2D_DESC texDesc = {
        .Width          = ctx->width,
        .Height         = ctx->height,
        .MipLevels      = 1,
        .Format         = format,
        .SampleDesc     = { .Count = 1 },
        .ArraySize      = 1,
        .Usage          = D3D11_USAGE_STAGING,
        .CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE,
    };

    hr = ID3D11Device_CreateTexture2D(device_hwctx->device, &texDesc, NULL, &s->staging_texture);
    if (FAILED(hr)) {
        av_log(ctx, AV_LOG_ERROR, "Could not create the staging texture (%lx)\n", (long)hr);
        return AVERROR_UNKNOWN;
    }

    return 0;
}

static void fill_texture_ptrs(uint8_t *data[4], int linesize[4],
                              AVHWFramesContext *ctx,
                              D3D11_TEXTURE2D_DESC *desc,
                              D3D11_MAPPED_SUBRESOURCE *map)
{
    int i;

    for (i = 0; i < 4; i++)
        linesize[i] = map->RowPitch;

    av_image_fill_pointers(data, ctx->sw_format, desc->Height,
                           (uint8_t*)map->pData, linesize);
}

static int d3d11va_transfer_data(AVHWFramesContext *ctx, AVFrame *dst,
                                 const AVFrame *src)
{
    AVD3D11VADeviceContext *device_hwctx = ctx->device_ctx->hwctx;
    D3D11VAFramesContext              *s = ctx->hwctx;
    int download = src->format == AV_PIX_FMT_D3D11;
    const AVFrame *frame = download ? src : dst;
    const AVFrame *other = download ? dst : src;
    // (The interface types are compatible.)
    ID3D11Resource *texture = (ID3D11Resource *)(ID3D11Texture2D *)frame->data[0];
    int index = (intptr_t)frame->data[1];
    ID3D11Resource *staging;
    int w = FFMIN(dst->width,  src->width);
    int h = FFMIN(dst->height, src->height);
    uint8_t *map_data[4];
    int map_linesize[4];
    D3D11_TEXTURE2D_DESC desc;
    D3D11_MAPPED_SUBRESOURCE map;
    HRESULT hr;
    int res;

    if (frame->hw_frames_ctx->data != (uint8_t *)ctx)
        return AVERROR(EINVAL);

#if CONFIG_CUDA
    if (other->format == AV_PIX_FMT_CUDA)
        return d3d11va_cuda_transfer_data(ctx, dst, src);
#endif

    /* Not a transfer to or from a software frame we can handle. Report this as
     * unimplemented rather than invalid, so that a hardware to hardware
     * transfer can still be tried from the other side. */
    if (other->format != ctx->sw_format)
        return AVERROR(ENOSYS);

    device_hwctx->lock(device_hwctx->lock_ctx);

    if (!s->staging_texture) {
        ID3D11Texture2D_GetDesc((ID3D11Texture2D *)texture, &desc);
        res = d3d11va_create_staging_texture(ctx, desc.Format);
        if (res < 0)
            return res;
    }

    staging = (ID3D11Resource *)s->staging_texture;

    ID3D11Texture2D_GetDesc(s->staging_texture, &desc);

    if (download) {
        ID3D11DeviceContext_CopySubresourceRegion(device_hwctx->device_context,
                                                  staging, 0, 0, 0, 0,
                                                  texture, index, NULL);

        hr = ID3D11DeviceContext_Map(device_hwctx->device_context,
                                     staging, 0, D3D11_MAP_READ, 0, &map);
        if (FAILED(hr))
            goto map_failed;

        fill_texture_ptrs(map_data, map_linesize, ctx, &desc, &map);

        av_image_copy2(dst->data, dst->linesize, map_data, map_linesize,
                       ctx->sw_format, w, h);

        ID3D11DeviceContext_Unmap(device_hwctx->device_context, staging, 0);
    } else {
        hr = ID3D11DeviceContext_Map(device_hwctx->device_context,
                                     staging, 0, D3D11_MAP_WRITE, 0, &map);
        if (FAILED(hr))
            goto map_failed;

        fill_texture_ptrs(map_data, map_linesize, ctx, &desc, &map);

        av_image_copy2(map_data, map_linesize, src->data, src->linesize,
                       ctx->sw_format, w, h);

        ID3D11DeviceContext_Unmap(device_hwctx->device_context, staging, 0);

        ID3D11DeviceContext_CopySubresourceRegion(device_hwctx->device_context,
                                                  texture, index, 0, 0, 0,
                                                  staging, 0, NULL);
    }

    device_hwctx->unlock(device_hwctx->lock_ctx);
    return 0;

map_failed:
    av_log(ctx, AV_LOG_ERROR, "Unable to lock D3D11VA surface (%lx)\n", (long)hr);
    device_hwctx->unlock(device_hwctx->lock_ctx);
    return AVERROR_UNKNOWN;
}

#if CONFIG_VULKAN || CONFIG_CUDA

/* The plane bridge shared by the interop transfer paths, see
 * hwcontext_d3d11va_internal.h for what it is and why. */

static const char bridge_shader_r[] =
    "Texture2D<float>   s : register(t0);\n"
    "RWTexture2D<float> d : register(u0);\n"
    "[numthreads(8, 8, 1)]\n"
    "void main(uint3 t : SV_DispatchThreadID) { d[t.xy] = s[t.xy]; }\n";

static const char bridge_shader_rg[] =
    "Texture2D<float2>   s : register(t0);\n"
    "RWTexture2D<float2> d : register(u0);\n"
    "[numthreads(8, 8, 1)]\n"
    "void main(uint3 t : SV_DispatchThreadID) { d[t.xy] = s[t.xy]; }\n";

int ff_d3d11va_bridge_formats(enum AVPixelFormat sw, DXGI_FORMAT *tex,
                              DXGI_FORMAT plane[2])
{
    switch (sw) {
    case AV_PIX_FMT_NV12:
        *tex = DXGI_FORMAT_NV12;
        plane[0] = DXGI_FORMAT_R8_UNORM;
        plane[1] = DXGI_FORMAT_R8G8_UNORM;
        return 0;
    case AV_PIX_FMT_P010:
    case AV_PIX_FMT_P012:
    case AV_PIX_FMT_P016:
        *tex = sw == AV_PIX_FMT_P010 ? DXGI_FORMAT_P010 : DXGI_FORMAT_P016;
        plane[0] = DXGI_FORMAT_R16_UNORM;
        plane[1] = DXGI_FORMAT_R16G16_UNORM;
        return 0;
    default:
        return AVERROR(ENOSYS);
    }
}

void ff_d3d11va_bridge_free(FFD3D11PlaneBridge **bridge)
{
    FFD3D11PlaneBridge *b = *bridge;

    if (!b)
        return;
    for (int i = 0; i < 2; i++) {
        if (b->plane_uav[i])
            ID3D11UnorderedAccessView_Release(b->plane_uav[i]);
        if (b->plane_srv[i])
            ID3D11ShaderResourceView_Release(b->plane_srv[i]);
        if (b->planes[i])
            ID3D11Texture2D_Release(b->planes[i]);
        if (b->staging_uav[i])
            ID3D11UnorderedAccessView_Release(b->staging_uav[i]);
        if (b->staging_srv[i])
            ID3D11ShaderResourceView_Release(b->staging_srv[i]);
        if (b->cs[i])
            ID3D11ComputeShader_Release(b->cs[i]);
    }
    if (b->staging)
        ID3D11Texture2D_Release(b->staging);
    if (b->compiler)
        dlclose(b->compiler);
    av_freep(bridge);
}

int ff_d3d11va_hr_err(HRESULT hr)
{
    return hr == E_OUTOFMEMORY ? AVERROR(ENOMEM) : AVERROR(ENOSYS);
}

int ff_d3d11va_texture_format(enum AVPixelFormat sw_format,
                              DXGI_FORMAT *format)
{
    for (int i = 0; i < FF_ARRAY_ELEMS(supported_formats); i++) {
        if (supported_formats[i].pix_fmt == sw_format) {
            *format = supported_formats[i].d3d_format;
            return 0;
        }
    }
    return AVERROR(ENOSYS);
}

int ff_d3d11va_shared_texture_create(ID3D11Device *dev, int width, int height,
                                     DXGI_FORMAT format, UINT bind_flags,
                                     ID3D11Texture2D **tex, void *log_ctx)
{
    D3D11_TEXTURE2D_DESC desc = {
        .Width      = width,
        .Height     = height,
        .MipLevels  = 1,
        .ArraySize  = 1,
        .Format     = format,
        .SampleDesc = { .Count = 1 },
        .Usage      = D3D11_USAGE_DEFAULT,
        .BindFlags  = bind_flags,
        .MiscFlags  = D3D11_RESOURCE_MISC_SHARED |
                      D3D11_RESOURCE_MISC_SHARED_NTHANDLE,
    };
    HRESULT hr;

    *tex = NULL;
    hr = ID3D11Device_CreateTexture2D(dev, &desc, NULL, tex);
    /* D3D11.0 runtimes reject NT handle sharing. Running out of memory says
     * nothing about the runtime, so it does not fall back. */
    if (FAILED(hr) && hr != E_OUTOFMEMORY) {
        desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
        hr = ID3D11Device_CreateTexture2D(dev, &desc, NULL, tex);
    }
    if (FAILED(hr)) {
        av_log(log_ctx, AV_LOG_DEBUG, "Cannot create a shared texture (%lx)\n",
               (long)hr);
        *tex = NULL;
        return ff_d3d11va_hr_err(hr);
    }
    return 0;
}

int ff_d3d11va_bridge_create(FFD3D11PlaneBridge **bridge, ID3D11Device *dev,
                             int width, int height,
                             enum AVPixelFormat sw_format, void *log_ctx)
{
    const char *cs_src[2] = { bridge_shader_r, bridge_shader_rg };
    const AVPixFmtDescriptor *pixdesc = av_pix_fmt_desc_get(sw_format);
    D3D11_TEXTURE2D_DESC desc;
    DXGI_FORMAT fmt, pfmts[2];
    FFD3D11PlaneBridge *b;
    pD3DCompile compile;
    HRESULT hr;
    int err;

    err = ff_d3d11va_bridge_formats(sw_format, &fmt, pfmts);
    if (err < 0)
        return err;

    b = *bridge = av_mallocz(sizeof(*b));
    if (!b)
        return AVERROR(ENOMEM);
    b->width      = width;
    b->height     = height;
    b->plane_w[0] = width;
    b->plane_h[0] = height;
    b->plane_w[1] = AV_CEIL_RSHIFT(width,  pixdesc->log2_chroma_w);
    b->plane_h[1] = AV_CEIL_RSHIFT(height, pixdesc->log2_chroma_h);

    /* The shaders are compiled at run time through d3dcompiler_47, loaded
     * on demand: libavutil cannot compile HLSL at build time, and a system
     * without the DLL keeps the system memory route. */
    b->compiler = dlopen("d3dcompiler_47.dll", 0);
    compile = b->compiler ? (pD3DCompile)dlsym(b->compiler, "D3DCompile")
                          : NULL;
    if (!compile) {
        av_log(log_ctx, AV_LOG_DEBUG, "d3dcompiler_47 is not available\n");
        err = AVERROR(ENOSYS);
        goto fail;
    }

    for (int i = 0; i < 2; i++) {
        ID3DBlob *code = NULL, *errors = NULL;
        hr = compile(cs_src[i], strlen(cs_src[i]), NULL, NULL, NULL, "main",
                     "cs_5_0", 0, 0, &code, &errors);
        if (SUCCEEDED(hr))
            hr = ID3D11Device_CreateComputeShader(dev,
                    ID3D10Blob_GetBufferPointer(code),
                    ID3D10Blob_GetBufferSize(code), NULL, &b->cs[i]);
        if (code)
            ID3D10Blob_Release(code);
        if (errors)
            ID3D10Blob_Release(errors);
        if (FAILED(hr)) {
            av_log(log_ctx, AV_LOG_DEBUG,
                   "Cannot build the copy shader (%lx)\n", (long)hr);
            err = ff_d3d11va_hr_err(hr);
            goto fail;
        }
    }

    desc = (D3D11_TEXTURE2D_DESC) {
        .Width      = width,
        .Height     = height,
        .MipLevels  = 1,
        .ArraySize  = 1,
        .Format     = fmt,
        .SampleDesc = { .Count = 1 },
        .Usage      = D3D11_USAGE_DEFAULT,
        .BindFlags  = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS,
    };
    hr = ID3D11Device_CreateTexture2D(dev, &desc, NULL, &b->staging);
    if (FAILED(hr) && hr != E_OUTOFMEMORY) {
        /* Without UAV support the plane textures cannot be assembled into the
         * staging texture, but moves toward them only ever read it, so keep
         * those working. Running out of memory says nothing about UAV
         * support, so it does not fall back. */
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        hr = ID3D11Device_CreateTexture2D(dev, &desc, NULL, &b->staging);
    }
    if (FAILED(hr)) {
        av_log(log_ctx, AV_LOG_DEBUG,
               "Cannot create the staging texture (%lx)\n", (long)hr);
        err = ff_d3d11va_hr_err(hr);
        goto fail;
    }

    for (int i = 0; i < 2; i++) {
        D3D11_SHADER_RESOURCE_VIEW_DESC sd = {
            .ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D,
            .Texture2D     = { .MipLevels = 1 },
        };
        D3D11_UNORDERED_ACCESS_VIEW_DESC ud = {
            .ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D,
        };

        /* A view with a single-plane format selects that plane of the planar
         * staging texture. */
        sd.Format = pfmts[i];
        ud.Format = pfmts[i];
        hr = ID3D11Device_CreateShaderResourceView(dev,
                                                   (ID3D11Resource *)b->staging,
                                                   &sd, &b->staging_srv[i]);
        if (FAILED(hr)) {
            av_log(log_ctx, AV_LOG_DEBUG, "Cannot create a plane view (%lx)\n",
                   (long)hr);
            err = ff_d3d11va_hr_err(hr);
            goto fail;
        }
        if (desc.BindFlags & D3D11_BIND_UNORDERED_ACCESS) {
            /* A view the format does not support leaves downloads
             * unsupported; running out of memory is worth a retry. */
            hr = ID3D11Device_CreateUnorderedAccessView(dev,
                    (ID3D11Resource *)b->staging, &ud, &b->staging_uav[i]);
            if (hr == E_OUTOFMEMORY) {
                err = AVERROR(ENOMEM);
                goto fail;
            }
            if (FAILED(hr))
                av_log(log_ctx, AV_LOG_DEBUG,
                       "No staging view for plane %d (%lx), no downloads\n",
                       i, (long)hr);
        }

        err = ff_d3d11va_shared_texture_create(dev, b->plane_w[i],
                                               b->plane_h[i], pfmts[i],
                                               D3D11_BIND_SHADER_RESOURCE |
                                               D3D11_BIND_UNORDERED_ACCESS,
                                               &b->planes[i], log_ctx);
        if (err < 0)
            goto fail;
        hr = ID3D11Device_CreateShaderResourceView(dev,
                (ID3D11Resource *)b->planes[i], NULL, &b->plane_srv[i]);
        if (SUCCEEDED(hr))
            hr = ID3D11Device_CreateUnorderedAccessView(dev,
                    (ID3D11Resource *)b->planes[i], NULL, &b->plane_uav[i]);
        if (FAILED(hr)) {
            av_log(log_ctx, AV_LOG_DEBUG, "Cannot create a plane view (%lx)\n",
                   (long)hr);
            err = ff_d3d11va_hr_err(hr);
            goto fail;
        }
    }

    /* Assembling writes every staging plane, so all or nothing. */
    if (!b->staging_uav[0] || !b->staging_uav[1]) {
        for (int i = 0; i < 2; i++) {
            if (b->staging_uav[i])
                ID3D11UnorderedAccessView_Release(b->staging_uav[i]);
            b->staging_uav[i] = NULL;
        }
    }

    return 0;

fail:
    ff_d3d11va_bridge_free(bridge);
    return err;
}

void ff_d3d11va_bridge_run(FFD3D11PlaneBridge *b, ID3D11DeviceContext *ctx,
                           ID3D11Resource *tex, unsigned index, int to_planes)
{
    ID3D11ShaderResourceView *null_srv = NULL;
    ID3D11UnorderedAccessView *null_uav = NULL;
    ID3D11ComputeShader *prev_cs = NULL;
    ID3D11ClassInstance *prev_inst[D3D11_SHADER_MAX_INTERFACES];
    UINT prev_inst_n = FF_ARRAY_ELEMS(prev_inst);
    ID3D11ShaderResourceView *prev_srv = NULL;
    ID3D11UnorderedAccessView *prev_uav = NULL;
    D3D11_BOX box = { 0, 0, 0, b->width, b->height, 1 };

    /* The context belongs to the caller, so everything the passes below bind
     * is saved here and put back at the end. */
    ID3D11DeviceContext_CSGetShader(ctx, &prev_cs, prev_inst, &prev_inst_n);
    ID3D11DeviceContext_CSGetShaderResources(ctx, 0, 1, &prev_srv);
    ID3D11DeviceContext_CSGetUnorderedAccessViews(ctx, 0, 1, &prev_uav);

    if (to_planes)
        ID3D11DeviceContext_CopySubresourceRegion(ctx,
            (ID3D11Resource *)b->staging, 0, 0, 0, 0, tex, index, &box);

    for (int i = 0; i < 2; i++) {
        ID3D11DeviceContext_CSSetShader(ctx, b->cs[i], NULL, 0);
        ID3D11DeviceContext_CSSetShaderResources(ctx, 0, 1,
            to_planes ? &b->staging_srv[i] : &b->plane_srv[i]);
        ID3D11DeviceContext_CSSetUnorderedAccessViews(ctx, 0, 1,
            to_planes ? &b->plane_uav[i] : &b->staging_uav[i], NULL);
        ID3D11DeviceContext_Dispatch(ctx, (b->plane_w[i] + 7) / 8,
                                     (b->plane_h[i] + 7) / 8, 1);
        /* Unbind before the next pass: two views of one resource cannot be
         * bound as input and output at the same time. */
        ID3D11DeviceContext_CSSetShaderResources(ctx, 0, 1, &null_srv);
        ID3D11DeviceContext_CSSetUnorderedAccessViews(ctx, 0, 1, &null_uav,
                                                      NULL);
    }

    ID3D11DeviceContext_CSSetShader(ctx, prev_cs, prev_inst, prev_inst_n);
    ID3D11DeviceContext_CSSetShaderResources(ctx, 0, 1, &prev_srv);
    ID3D11DeviceContext_CSSetUnorderedAccessViews(ctx, 0, 1, &prev_uav, NULL);
    if (prev_cs)
        ID3D11ComputeShader_Release(prev_cs);
    for (UINT i = 0; i < prev_inst_n; i++)
        ID3D11ClassInstance_Release(prev_inst[i]);
    if (prev_srv)
        ID3D11ShaderResourceView_Release(prev_srv);
    if (prev_uav)
        ID3D11UnorderedAccessView_Release(prev_uav);

    if (!to_planes)
        ID3D11DeviceContext_CopySubresourceRegion(ctx, tex, index, 0, 0, 0,
            (ID3D11Resource *)b->staging, 0, NULL);
}

#endif /* CONFIG_VULKAN || CONFIG_CUDA */

#if CONFIG_CUDA

/* ffnvcodec does not carry these yet; the values come from cuda.h. The
 * FF_ prefix keeps them from clashing with the real definitions once
 * ffnvcodec has them. */
#define FF_CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE \
    ((CUexternalMemoryHandleType)6)
#define FF_CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE \
    ((CUexternalSemaphoreHandleType)5)
#define FF_CUDA_EXTERNAL_MEMORY_DEDICATED 0x1
#define FF_CU_AD_FORMAT_UNORM_INT_101010_2 ((CUarray_format)0x50)
#define FF_CUDA_ERROR_INVALID_VALUE   ((CUresult)1)
#define FF_CUDA_ERROR_OUT_OF_MEMORY   ((CUresult)2)
#define FF_CUDA_ERROR_INVALID_HANDLE  ((CUresult)400)
#define FF_CUDA_ERROR_NOT_SUPPORTED   ((CUresult)801)

/* CUDA shares a D3D11 texture through the external memory API: each of the
 * private textures below is imported once as a CUDA mipmapped array, and a
 * shared D3D11 fence, imported as a CUDA external semaphore, chains the two
 * queues on the GPU in both directions, so no transfer ever waits on the
 * CPU. What no CUDA sharing API can do is reach the second plane of a
 * two-plane texture, so those frames move through the plane bridge, and
 * single-plane frames through an intermediate texture in their own format,
 * so the frame textures themselves, which may be recycled decoder arrays,
 * never have to be imported. One state is kept for each CUDA context frames
 * have been transferred to or from, created once and kept; cuda_lock
 * serializes the transfers, which keeps the shared textures owned by one
 * transfer at a time and the fence values increasing in submission order,
 * which monitored fences require of their signals. The copies ride the
 * CUDA device's stream, so the D3D11 waits they end in also depend on
 * whatever else the application keeps on that stream. */
typedef struct D3D11CudaInterop {
    struct D3D11CudaInterop *next;
    AVBufferRef        *device_ref; /* keeps the loader and the stream alive */
    CUcontext           cuda_ctx;   /* identity of the pairing */
    int                 status;     /* 0 untried, 1 ready, else error */
    FFD3D11PlaneBridge *bridge;     /* two-plane formats */
    ID3D11Texture2D    *tex;        /* single-plane intermediate */
    ID3D11DeviceContext4 *ctx4;
    ID3D11Fence        *fence;
    CUexternalSemaphore sem;
    CUstream            stream;     /* the one that set cuda_done */
    CUexternalMemory    mem[2];
    CUmipmappedArray    mip[2];
    CUarray             arr[2];     /* level 0 of each texture */
    int                 nb_planes;
    uint64_t            fence_val;  /* last fence value handed out */
    uint64_t            cuda_done;  /* last value CUDA was told to signal */
} D3D11CudaInterop;

/* ffnvcodec loads the external memory and semaphore entry points
 * optionally, so a loader built against an older CUDA can leave them NULL.
 * They are all checked before any interop state is created, so a teardown
 * never has to cope with imports that could not have been made. */
static int d3d11va_cuda_interop_available(CudaFunctions *cu)
{
    return cu->cuImportExternalMemory && cu->cuDestroyExternalMemory &&
           cu->cuExternalMemoryGetMappedMipmappedArray &&
           cu->cuMipmappedArrayGetLevel && cu->cuMipmappedArrayDestroy &&
           cu->cuImportExternalSemaphore && cu->cuDestroyExternalSemaphore &&
           cu->cuSignalExternalSemaphoresAsync &&
           cu->cuWaitExternalSemaphoresAsync;
}

/* Only a lack of interoperability, reported as ENOSYS, condemns the pairing
 * below. Running out of memory is worth retrying, and the import functions
 * are documented to fail for reasons unrelated to what is being imported,
 * an operating system call or an earlier asynchronous error, which say
 * nothing about the pairing either. */
static int d3d11va_cuda_cu_err(CUresult ret)
{
    switch (ret) {
    case FF_CUDA_ERROR_OUT_OF_MEMORY:
        return AVERROR(ENOMEM);
    case FF_CUDA_ERROR_INVALID_VALUE:
    case FF_CUDA_ERROR_INVALID_HANDLE:
    case FF_CUDA_ERROR_NOT_SUPPORTED:
        return AVERROR(ENOSYS);
    default:
        return AVERROR_EXTERNAL;
    }
}

/* Release everything a pairing created, imports included, leaving the
 * pairing itself in place. */
static void d3d11va_cuda_interop_release(AVHWFramesContext *ctx,
                                         D3D11CudaInterop *ci)
{
    AVHWDeviceContext *dev_ctx = (AVHWDeviceContext *)ci->device_ref->data;
    AVCUDADeviceContext *cu_hw = dev_ctx->hwctx;
    CudaFunctions *cu = cu_hw->internal->cuda_dl;
    CUcontext dummy;

    if (ci->sem || ci->mem[0] || ci->mem[1]) {
        if (cu->cuCtxPushCurrent(ci->cuda_ctx) == CUDA_SUCCESS) {
            /* Transfers do not wait for their copies, so make sure none is
             * still using the imports about to go away. Every transfer
             * opens with a D3D11 wait for the value the previous one
             * signals, and its copies only run after the signal that
             * follows that wait, so they are ordered behind all earlier
             * copies whichever streams those used. Synchronizing the last
             * stream therefore drains them all. That stream is NULL for the
             * default stream, so what tells a transfer happened is the
             * value it signaled. */
            if (ci->cuda_done)
                cu->cuStreamSynchronize(ci->stream);
            if (ci->sem)
                cu->cuDestroyExternalSemaphore(ci->sem);
            for (int i = 0; i < FF_ARRAY_ELEMS(ci->mem); i++) {
                if (ci->mip[i])
                    cu->cuMipmappedArrayDestroy(ci->mip[i]);
                if (ci->mem[i])
                    cu->cuDestroyExternalMemory(ci->mem[i]);
            }
            cu->cuCtxPopCurrent(&dummy);
        } else {
            av_log(ctx, AV_LOG_WARNING, "The CUDA context is gone; its "
                   "imports leak, and in-flight copies cannot be waited "
                   "out\n");
        }
    }
    ci->sem = NULL;
    memset(ci->mem, 0, sizeof(ci->mem));
    memset(ci->mip, 0, sizeof(ci->mip));
    memset(ci->arr, 0, sizeof(ci->arr));
    ci->stream    = NULL;
    ci->cuda_done = 0;
    ci->fence_val = 0;
    ci->nb_planes = 0;
    ff_d3d11va_bridge_free(&ci->bridge);
    if (ci->tex)
        ID3D11Texture2D_Release(ci->tex);
    ci->tex = NULL;
    if (ci->fence)
        ID3D11Fence_Release(ci->fence);
    ci->fence = NULL;
    if (ci->ctx4)
        ID3D11DeviceContext4_Release(ci->ctx4);
    ci->ctx4 = NULL;
}

static void d3d11va_cuda_interops_free(AVHWFramesContext *ctx)
{
    D3D11VAFramesContext *s = ctx->hwctx;
    D3D11CudaInterop *ci = s->cuda_interop;

    while (ci) {
        D3D11CudaInterop *next = ci->next;
        d3d11va_cuda_interop_release(ctx, ci);
        av_buffer_unref(&ci->device_ref);
        av_free(ci);
        ci = next;
    }
    s->cuda_interop = NULL;
}

/* The CUDA array layout of each texture format: an import has to describe
 * the texture the way the exporting API does. */
static const struct {
    DXGI_FORMAT d3d_format;
    CUarray_format format;
    int channels;
    int texel; /* bytes */
} cuda_array_formats[] = {
    { DXGI_FORMAT_R8_UNORM, CU_AD_FORMAT_UNSIGNED_INT8, 1, 1 },
    { DXGI_FORMAT_R8G8_UNORM, CU_AD_FORMAT_UNSIGNED_INT8, 2, 2 },
    { DXGI_FORMAT_R16_UNORM, CU_AD_FORMAT_UNSIGNED_INT16, 1, 2 },
    { DXGI_FORMAT_R16G16_UNORM, CU_AD_FORMAT_UNSIGNED_INT16, 2, 4 },
    { DXGI_FORMAT_B8G8R8A8_UNORM, CU_AD_FORMAT_UNSIGNED_INT8, 4, 4 },
    { DXGI_FORMAT_R8G8B8A8_UNORM, CU_AD_FORMAT_UNSIGNED_INT8, 4, 4 },
    { DXGI_FORMAT_R10G10B10A2_UNORM, FF_CU_AD_FORMAT_UNORM_INT_101010_2, 4, 4 },
    { DXGI_FORMAT_R16G16B16A16_FLOAT, CU_AD_FORMAT_HALF, 4, 8 },
};

/* Import one of our own textures, which is NT handle shared, as a CUDA
 * array. */
static int d3d11va_cuda_import_texture(CudaFunctions *cu,
                                       ID3D11Texture2D *tex,
                                       CUexternalMemory *mem,
                                       CUmipmappedArray *mip, CUarray *arr)
{
    D3D11_TEXTURE2D_DESC desc;
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC mdesc;
    CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC adesc;
    CUarray_format format = 0;
    int channels = 0, texel = 0;
    IDXGIResource1 *res1;
    HANDLE handle;
    HRESULT hr;
    CUresult ret;

    ID3D11Texture2D_GetDesc(tex, &desc);
    for (int i = 0; i < FF_ARRAY_ELEMS(cuda_array_formats); i++) {
        if (cuda_array_formats[i].d3d_format == desc.Format) {
            format   = cuda_array_formats[i].format;
            channels = cuda_array_formats[i].channels;
            texel    = cuda_array_formats[i].texel;
            break;
        }
    }
    /* KMT handles import into CUDA without an error, and then every access
     * faults. */
    if (!channels || !(desc.MiscFlags & D3D11_RESOURCE_MISC_SHARED_NTHANDLE))
        return AVERROR(ENOSYS);

    /* D3D11 never reports the size of a texture's allocation, and NVIDIA
     * documents no value for D3D11 resources, so the size given is that of
     * the texels, tightly packed: a lower bound, not the real size. A
     * dedicated import is bound to the resource, whose real allocation the
     * driver can see, and the NVIDIA driver, the only implementation, has
     * accepted this size on every driver this code was tested with. */
    mdesc = (CUDA_EXTERNAL_MEMORY_HANDLE_DESC) {
        .type  = FF_CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE,
        .size  = (uint64_t)desc.Width * desc.Height * texel,
        .flags = FF_CUDA_EXTERNAL_MEMORY_DEDICATED,
    };
    hr = ID3D11Texture2D_QueryInterface(tex, &IID_IDXGIResource1,
                                        (void **)&res1);
    if (FAILED(hr))
        return AVERROR(ENOSYS);
    hr = IDXGIResource1_CreateSharedHandle(res1, NULL,
                                           DXGI_SHARED_RESOURCE_READ |
                                           DXGI_SHARED_RESOURCE_WRITE,
                                           NULL, &handle);
    IDXGIResource1_Release(res1);
    /* The handle says nothing about what the pairing can do, so its failure
     * is not remembered, unlike the import's below. */
    if (FAILED(hr))
        return hr == E_OUTOFMEMORY ? AVERROR(ENOMEM) : AVERROR_EXTERNAL;
    mdesc.handle.win32.handle = handle;
    ret = cu->cuImportExternalMemory(mem, &mdesc);
    CloseHandle(handle);
    if (ret != CUDA_SUCCESS)
        return d3d11va_cuda_cu_err(ret);

    adesc = (CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC) {
        .arrayDesc = {
            .Width       = desc.Width,
            .Height      = desc.Height,
            .Format      = format,
            .NumChannels = channels,
        },
        .numLevels = 1,
    };
    ret = cu->cuExternalMemoryGetMappedMipmappedArray(mip, *mem, &adesc);
    if (ret != CUDA_SUCCESS)
        return d3d11va_cuda_cu_err(ret);

    ret = cu->cuMipmappedArrayGetLevel(arr, *mip, 0);
    if (ret != CUDA_SUCCESS)
        return d3d11va_cuda_cu_err(ret);
    return 0;
}

/* The imports below need the CUDA context on the adapter the D3D11 device
 * runs on, and the driver documents no particular error for a handle from
 * another one, so the two are compared up front, by LUID like the device
 * derivation does. Returns 1 when they are known to differ; when either
 * LUID cannot be obtained, the import itself has to tell. */
static int d3d11va_cuda_other_adapter(AVHWFramesContext *ctx,
                                      AVCUDADeviceContext *cu_hw)
{
    AVD3D11VADeviceContext *hwctx = ctx->device_ctx->hwctx;
    IDXGIDevice *dxgi_dev;
    IDXGIAdapter *adapter;
    DXGI_ADAPTER_DESC desc;
    LUID luid;
    HRESULT hr;

    if (d3d11va_cuda_luid(cu_hw, &luid) < 0)
        return 0;

    hr = ID3D11Device_QueryInterface(hwctx->device, &IID_IDXGIDevice,
                                     (void **)&dxgi_dev);
    if (FAILED(hr))
        return 0;
    hr = IDXGIDevice_GetAdapter(dxgi_dev, &adapter);
    IDXGIDevice_Release(dxgi_dev);
    if (FAILED(hr))
        return 0;
    hr = IDXGIAdapter_GetDesc(adapter, &desc);
    IDXGIAdapter_Release(adapter);
    if (FAILED(hr))
        return 0;

    return desc.AdapterLuid.LowPart  != luid.LowPart ||
           desc.AdapterLuid.HighPart != luid.HighPart;
}

/* Find or create the state for this CUDA context, cuda_lock held. An
 * attempt that found the two unable to share is kept, so it is not retried
 * every frame; any other failure releases what the attempt created, and
 * the next transfer starts over. */
static int d3d11va_cuda_interop_get(AVHWFramesContext *ctx,
                                    AVHWFramesContext *cuda_fc,
                                    D3D11CudaInterop **out)
{
    D3D11VAFramesContext *s = ctx->hwctx;
    AVD3D11VADeviceContext *hwctx = ctx->device_ctx->hwctx;
    AVCUDADeviceContext *cu_hw = cuda_fc->device_ctx->hwctx;
    CudaFunctions *cu = cu_hw->internal->cuda_dl;
    const int planes = av_pix_fmt_count_planes(ctx->sw_format);
    ID3D11Texture2D *plane_tex[2];
    D3D11CudaInterop *ci;
    ID3D11Device5 *dev5;
    CUcontext dummy;
    HANDLE handle;
    HRESULT hr;
    int err;

    for (ci = s->cuda_interop; ci; ci = ci->next)
        if (ci->cuda_ctx == cu_hw->cuda_ctx)
            break;

    if (!ci) {
        ci = av_mallocz(sizeof(*ci));
        if (!ci)
            return AVERROR(ENOMEM);
        ci->cuda_ctx   = cu_hw->cuda_ctx;
        ci->device_ref = av_buffer_ref(cuda_fc->device_ref);
        if (!ci->device_ref) {
            av_free(ci);
            return AVERROR(ENOMEM);
        }
        ci->next = s->cuda_interop;
        s->cuda_interop = ci;
    }
    if (ci->status) {
        *out = ci;
        return ci->status > 0 ? 0 : ci->status;
    }

    if (d3d11va_cuda_other_adapter(ctx, cu_hw)) {
        av_log(ctx, AV_LOG_DEBUG, "The CUDA context is on another adapter\n");
        err = AVERROR(ENOSYS);
        goto fail;
    }

    if (planes == 2) {
        err = ff_d3d11va_bridge_create(&ci->bridge, hwctx->device,
                                       ctx->width, ctx->height,
                                       ctx->sw_format, ctx);
        if (err < 0)
            goto fail;
        plane_tex[0] = ci->bridge->planes[0];
        plane_tex[1] = ci->bridge->planes[1];
        ci->nb_planes = 2;
    } else {
        err = ff_d3d11va_shared_texture_create(hwctx->device, ctx->width,
                                               ctx->height, s->format, 0,
                                               &ci->tex, ctx);
        if (err < 0)
            goto fail;
        plane_tex[0] = ci->tex;
        ci->nb_planes = 1;
    }

    hr = ID3D11Device_QueryInterface(hwctx->device, &IID_ID3D11Device5,
                                     (void **)&dev5);
    if (SUCCEEDED(hr)) {
        hr = ID3D11Device5_CreateFence(dev5, 0, D3D11_FENCE_FLAG_SHARED,
                                       &IID_ID3D11Fence, (void **)&ci->fence);
        ID3D11Device5_Release(dev5);
    }
    if (FAILED(hr)) {
        av_log(ctx, AV_LOG_DEBUG, "Cannot create a shared fence (%lx)\n",
               (long)hr);
        err = ff_d3d11va_hr_err(hr);
        goto fail;
    }
    hwctx->lock(hwctx->lock_ctx);
    hr = ID3D11DeviceContext_QueryInterface(hwctx->device_context,
                                            &IID_ID3D11DeviceContext4,
                                            (void **)&ci->ctx4);
    hwctx->unlock(hwctx->lock_ctx);
    if (FAILED(hr)) {
        err = AVERROR(ENOSYS);
        goto fail;
    }

    /* The adapters were compared above where the driver allowed it, so
     * what the imports fail on is a pairing the two APIs cannot share,
     * reported as ENOSYS and remembered, or something transient, which the
     * next transfer retries. */
    if (cu->cuCtxPushCurrent(ci->cuda_ctx) != CUDA_SUCCESS) {
        err = AVERROR_EXTERNAL;
        goto fail;
    }
    hr = ID3D11Fence_CreateSharedHandle(ci->fence, NULL, GENERIC_ALL,
                                        NULL, &handle);
    if (SUCCEEDED(hr)) {
        CUresult ret;
        CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC sdesc = {
            .type = FF_CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE,
            .handle.win32.handle = handle,
        };
        ret = cu->cuImportExternalSemaphore(&ci->sem, &sdesc);
        err = ret == CUDA_SUCCESS ? 0 : d3d11va_cuda_cu_err(ret);
        CloseHandle(handle);
    } else {
        /* Like the texture handles: nothing the pairing can be blamed for. */
        err = hr == E_OUTOFMEMORY ? AVERROR(ENOMEM) : AVERROR_EXTERNAL;
    }
    for (int i = 0; err >= 0 && i < ci->nb_planes; i++)
        err = d3d11va_cuda_import_texture(cu, plane_tex[i], &ci->mem[i],
                                          &ci->mip[i], &ci->arr[i]);
    cu->cuCtxPopCurrent(&dummy);
    if (err < 0) {
        av_log(ctx, AV_LOG_DEBUG, "Cannot import a texture into CUDA\n");
        goto fail;
    }

    ci->status = 1;
    *out = ci;
    return 0;

fail:
    /* Whatever the attempt created is released, so that a retry starts
     * over, with textures whose sharing handles have not been created yet:
     * D3D11 documents a single creation per texture. Only a lack of
     * interoperability is worth remembering. Running out of memory, and
     * whatever the driver blames on something other than what is being
     * imported, may not repeat, so the next transfer retries. */
    d3d11va_cuda_interop_release(ctx, ci);
    if (err == AVERROR(ENOSYS))
        ci->status = err;
    *out = ci;
    return err;
}

static int d3d11va_cuda_transfer_data(AVHWFramesContext *ctx, AVFrame *dst,
                                      const AVFrame *src)
{
    D3D11VAFramesContext *s = ctx->hwctx;
    AVD3D11VADeviceContext *hwctx = ctx->device_ctx->hwctx;
    const int to_cuda = dst->format == AV_PIX_FMT_CUDA;
    const AVFrame *cudaf = to_cuda ? dst : src;
    const AVFrame *d3df  = to_cuda ? src : dst;
    ID3D11Resource *tex = (ID3D11Resource *)d3df->data[0];
    UINT index = (UINT)(intptr_t)d3df->data[1];
    const int planes = av_pix_fmt_count_planes(ctx->sw_format);
    const AVPixFmtDescriptor *pixdesc = av_pix_fmt_desc_get(ctx->sw_format);
    /* Like the system memory paths, only the common region is copied; with
     * differing sizes the margin keeps whatever the long-lived intermediate
     * textures held before, and those are sized by the frames context. */
    int w = FFMIN3(dst->width,  src->width,  ctx->width);
    int h = FFMIN3(dst->height, src->height, ctx->height);
    AVHWFramesContext *cuda_fc;
    AVHWDeviceContext *cuda_cu;
    AVCUDADeviceContext *cu_hw;
    CudaFunctions *cu;
    D3D11CudaInterop *ci;
    D3D11_TEXTURE2D_DESC desc;
    D3D11_BOX box = { 0, 0, 0, ctx->width, ctx->height, 1 };
    CUcontext dummy;
    uint64_t v1, v2;
    HRESULT hr;
    int ret;

    if (!cudaf->hw_frames_ctx)
        return AVERROR(ENOSYS);
    cuda_fc = (AVHWFramesContext *)cudaf->hw_frames_ctx->data;
    if (cuda_fc->format != AV_PIX_FMT_CUDA)
        return AVERROR(ENOSYS);
    cuda_cu = cuda_fc->device_ctx;
    cu_hw   = cuda_fc->device_ctx->hwctx;
    cu      = cu_hw->internal->cuda_dl;

    /* The copy moves bits, so the two sides have to agree on what the bits
     * mean, and there is no way to address the planes of more of them.
     * Packed subsampled formats are also out: their textures constrain the
     * copy regions in ways this code does not track. */
    if (cuda_fc->sw_format != ctx->sw_format || planes < 1 || planes > 2 ||
        (planes == 1 && (pixdesc->log2_chroma_w || pixdesc->log2_chroma_h)))
        return AVERROR(ENOSYS);

    /* The staging copies address the frame-sized region of the texture,
     * which decoders often pad, and copies of video formats only accept
     * aligned regions. They address the subresource by array slice, which
     * is only its index in a texture without mip levels. The texture must
     * really be in the format the sw format implies, or the copies would
     * silently move nothing. A keyed-mutex texture is only coherent for a
     * user that acquires the mutex, which this code does not do. */
    ID3D11Texture2D_GetDesc((ID3D11Texture2D *)tex, &desc);
    if (desc.Width < ctx->width || desc.Height < ctx->height ||
        desc.SampleDesc.Count != 1 || desc.MipLevels != 1 ||
        index >= desc.ArraySize ||
        (desc.MiscFlags & D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX))
        return AVERROR(ENOSYS);
    if (planes == 2) {
        DXGI_FORMAT tex_fmt, plane_fmts[2];
        if (((ctx->width | ctx->height) & 1) ||
            ff_d3d11va_bridge_formats(ctx->sw_format, &tex_fmt,
                                      plane_fmts) < 0 ||
            desc.Format != tex_fmt)
            return AVERROR(ENOSYS);
    } else {
        /* Formats no CUDA array layout matches are rejected before the
         * interop state allocates anything for them. */
        int i;
        for (i = 0; i < FF_ARRAY_ELEMS(cuda_array_formats); i++)
            if (cuda_array_formats[i].d3d_format == s->format)
                break;
        if (desc.Format != s->format ||
            i == FF_ARRAY_ELEMS(cuda_array_formats))
            return AVERROR(ENOSYS);
    }

    /* Without the optional entry points nothing can be imported, so no
     * interop state is created for this pairing either. */
    if (!d3d11va_cuda_interop_available(cu))
        return AVERROR(ENOSYS);

    ff_mutex_lock(&s->cuda_lock);

    ret = d3d11va_cuda_interop_get(ctx, cuda_fc, &ci);
    if (ret < 0)
        goto end;
    /* Assembling a frame writes the staging texture through views its format
     * does not support everywhere. */
    if (planes == 2 && !to_cuda && !ci->bridge->staging_uav[0]) {
        ret = AVERROR(ENOSYS);
        goto end;
    }

    v1 = ++ci->fence_val;
    v2 = ++ci->fence_val;

    hwctx->lock(hwctx->lock_ctx);
    /* The last transfer returned while its copies could still be touching
     * these textures; the wait orders whatever comes next after them, on
     * the GPU. Failures have to surface before anything crosses the APIs:
     * CUDA waiting on a signal that never got submitted would block its
     * stream for good. */
    hr = ID3D11DeviceContext4_Wait(ci->ctx4, ci->fence, ci->cuda_done);
    if (SUCCEEDED(hr) && to_cuda) {
        if (planes == 2)
            ff_d3d11va_bridge_run(ci->bridge, hwctx->device_context, tex,
                                  index, 1);
        else
            ID3D11DeviceContext_CopySubresourceRegion(hwctx->device_context,
                (ID3D11Resource *)ci->tex, 0, 0, 0, 0, tex, index, &box);
    }
    /* The signal orders all D3D11 work issued so far ahead of the CUDA
     * copies, whether it staged the frame or still reads the textures from
     * an earlier download; the flush submits it, or CUDA would wait on
     * work still sitting in the command buffer. */
    if (SUCCEEDED(hr))
        hr = ID3D11DeviceContext4_Signal(ci->ctx4, ci->fence, v1);
    ID3D11DeviceContext_Flush(hwctx->device_context);
    hwctx->unlock(hwctx->lock_ctx);
    if (FAILED(hr)) {
        ret = AVERROR_EXTERNAL;
        goto end;
    }

    ret = CHECK_CU(cu->cuCtxPushCurrent(ci->cuda_ctx));
    if (ret < 0)
        goto end;

    {
        CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS wait = {
            .params.fence.value = v1,
        };
        ret = CHECK_CU(cu->cuWaitExternalSemaphoresAsync(&ci->sem, &wait, 1,
                                                         cu_hw->stream));
    }

    for (int i = 0; ret >= 0 && i < planes; i++) {
        CUDA_MEMCPY2D cpy = {
            .WidthInBytes = av_image_get_linesize(ctx->sw_format, w, i),
            .Height       = i ? AV_CEIL_RSHIFT(h, pixdesc->log2_chroma_h) : h,
        };

        if (to_cuda) {
            cpy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
            cpy.srcArray      = ci->arr[i];
            cpy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            cpy.dstDevice     = (CUdeviceptr)(uintptr_t)cudaf->data[i];
            cpy.dstPitch      = cudaf->linesize[i];
        } else {
            cpy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            cpy.srcDevice     = (CUdeviceptr)(uintptr_t)cudaf->data[i];
            cpy.srcPitch      = cudaf->linesize[i];
            cpy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
            cpy.dstArray      = ci->arr[i];
        }
        ret = CHECK_CU(cu->cuMemcpy2DAsync(&cpy, cu_hw->stream));
    }

    if (ret >= 0) {
        CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS signal = {
            .params.fence.value = v2,
        };
        ret = CHECK_CU(cu->cuSignalExternalSemaphoresAsync(&ci->sem, &signal,
                                                           1, cu_hw->stream));
    }
    if (ret >= 0 && ci->device_ref->data != cuda_fc->device_ref->data) {
        /* The stream that has to be drained at teardown belongs to this
         * device context, which must not go away before it. */
        ret = av_buffer_replace(&ci->device_ref, cuda_fc->device_ref);
    }
    if (ret >= 0) {
        /* No synchronize: the copies stay ordered on the stream for CUDA
         * consumers, and behind the fence value for D3D11 ones. */
        ci->cuda_done = v2;
        ci->stream    = cu_hw->stream;
    } else {
        /* cuda_done was not advanced, so nothing will ever wait on v2,
         * whether it got signaled or not; the next transfer signals a
         * higher value, which is all a monitored fence asks. */
        CHECK_CU(cu->cuStreamSynchronize(cu_hw->stream));
    }
    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    if (ret < 0)
        goto end;

    if (!to_cuda) {
        /* Only past the fence value do the textures hold the frame; the
         * wait keeps the assembly behind them, on the GPU. If it cannot be
         * enqueued, the destination must not be assembled from stale
         * planes. */
        hwctx->lock(hwctx->lock_ctx);
        hr = ID3D11DeviceContext4_Wait(ci->ctx4, ci->fence, v2);
        if (SUCCEEDED(hr)) {
            if (planes == 2)
                ff_d3d11va_bridge_run(ci->bridge, hwctx->device_context, tex,
                                      index, 0);
            else
                ID3D11DeviceContext_CopySubresourceRegion(hwctx->device_context,
                    tex, index, 0, 0, 0, (ID3D11Resource *)ci->tex, 0, NULL);
        }
        hwctx->unlock(hwctx->lock_ctx);
        if (FAILED(hr)) {
            ret = AVERROR_EXTERNAL;
            goto end;
        }
    }
    ret = 0;

end:
    ff_mutex_unlock(&s->cuda_lock);
    return ret;
}

#endif /* CONFIG_CUDA */

static int d3d11va_device_init(AVHWDeviceContext *hwdev)
{
    AVD3D11VADeviceContext *device_hwctx = hwdev->hwctx;
    HRESULT hr;

    if (!device_hwctx->lock) {
        device_hwctx->lock_ctx = CreateMutex(NULL, 0, NULL);
        if (device_hwctx->lock_ctx == INVALID_HANDLE_VALUE) {
            av_log(NULL, AV_LOG_ERROR, "Failed to create a mutex\n");
            return AVERROR(EINVAL);
        }
        device_hwctx->lock   = d3d11va_default_lock;
        device_hwctx->unlock = d3d11va_default_unlock;
    }

    if (!device_hwctx->device_context) {
        ID3D11Device_GetImmediateContext(device_hwctx->device, &device_hwctx->device_context);
        if (!device_hwctx->device_context)
            return AVERROR_UNKNOWN;
    }

    if (!device_hwctx->video_device) {
        hr = ID3D11DeviceContext_QueryInterface(device_hwctx->device, &IID_ID3D11VideoDevice,
                                                (void **)&device_hwctx->video_device);
        if (FAILED(hr))
            return AVERROR_UNKNOWN;
    }

    if (!device_hwctx->video_context) {
        hr = ID3D11DeviceContext_QueryInterface(device_hwctx->device_context, &IID_ID3D11VideoContext,
                                                (void **)&device_hwctx->video_context);
        if (FAILED(hr))
            return AVERROR_UNKNOWN;
    }

    return 0;
}

static void d3d11va_device_uninit(AVHWDeviceContext *hwdev)
{
    AVD3D11VADeviceContext *device_hwctx = hwdev->hwctx;

    if (device_hwctx->device) {
        ID3D11Device_Release(device_hwctx->device);
        device_hwctx->device = NULL;
    }

    if (device_hwctx->device_context) {
        ID3D11DeviceContext_Release(device_hwctx->device_context);
        device_hwctx->device_context = NULL;
    }

    if (device_hwctx->video_device) {
        ID3D11VideoDevice_Release(device_hwctx->video_device);
        device_hwctx->video_device = NULL;
    }

    if (device_hwctx->video_context) {
        ID3D11VideoContext_Release(device_hwctx->video_context);
        device_hwctx->video_context = NULL;
    }

    if (device_hwctx->lock == d3d11va_default_lock) {
        CloseHandle(device_hwctx->lock_ctx);
        device_hwctx->lock_ctx = INVALID_HANDLE_VALUE;
        device_hwctx->lock = NULL;
    }
}

static int d3d11va_device_find_adapter_by_vendor_id(AVHWDeviceContext *ctx, uint32_t flags, const char *vendor_id)
{
    HRESULT hr;
    IDXGIAdapter *adapter = NULL;
    IDXGIFactory2 *factory;
    int adapter_id = 0;
    long int id = strtol(vendor_id, NULL, 0);

    hr = mCreateDXGIFactory(&IID_IDXGIFactory2, (void **)&factory);
    if (FAILED(hr)) {
        av_log(ctx, AV_LOG_ERROR, "CreateDXGIFactory returned error\n");
        return -1;
    }

    while (IDXGIFactory2_EnumAdapters(factory, adapter_id++, &adapter) != DXGI_ERROR_NOT_FOUND) {
        ID3D11Device* device = NULL;
        DXGI_ADAPTER_DESC adapter_desc;

        hr = mD3D11CreateDevice(adapter, D3D_DRIVER_TYPE_UNKNOWN, NULL, flags, NULL, 0, D3D11_SDK_VERSION, &device, NULL, NULL);
        if (FAILED(hr)) {
            av_log(ctx, AV_LOG_DEBUG, "D3D11CreateDevice returned error, try next adapter\n");
            IDXGIAdapter_Release(adapter);
            continue;
        }

        hr = IDXGIAdapter2_GetDesc(adapter, &adapter_desc);
        ID3D11Device_Release(device);
        IDXGIAdapter_Release(adapter);
        if (FAILED(hr)) {
            av_log(ctx, AV_LOG_DEBUG, "IDXGIAdapter2_GetDesc returned error, try next adapter\n");
            continue;
        } else if (adapter_desc.VendorId == id) {
            IDXGIFactory2_Release(factory);
            return adapter_id - 1;
        }
    }

    IDXGIFactory2_Release(factory);
    return -1;
}

static int d3d11va_device_create(AVHWDeviceContext *ctx, const char *device,
                                 AVDictionary *opts, int flags)
{
    AVD3D11VADeviceContext *device_hwctx = ctx->hwctx;

    HRESULT hr;
    IDXGIAdapter           *pAdapter = NULL;
    ID3D10Multithread      *pMultithread;
    UINT creationFlags = D3D11_CREATE_DEVICE_VIDEO_SUPPORT;
    int is_debug       = !!av_dict_get(opts, "debug", NULL, 0);
    int ret;
    int adapter = -1;

    if (is_debug) {
        creationFlags |= D3D11_CREATE_DEVICE_DEBUG;
        av_log(ctx, AV_LOG_INFO, "Enabling d3d11 debugging.\n");
    }

    if ((ret = ff_thread_once(&functions_loaded, load_functions)) != 0)
        return AVERROR_UNKNOWN;
    if (!mD3D11CreateDevice || !mCreateDXGIFactory) {
        av_log(ctx, AV_LOG_ERROR, "Failed to load D3D11 library or its functions\n");
        return AVERROR_UNKNOWN;
    }

    if (device) {
        adapter = atoi(device);
    } else {
        AVDictionaryEntry *e = av_dict_get(opts, "vendor_id", NULL, 0);
        if (e && e->value) {
            adapter = d3d11va_device_find_adapter_by_vendor_id(ctx, creationFlags, e->value);
            if (adapter < 0) {
                av_log(ctx, AV_LOG_ERROR, "Failed to find d3d11va adapter by "
                       "vendor id %s\n", e->value);
                return AVERROR_UNKNOWN;
            }
        }
    }

    if (adapter >= 0) {
        IDXGIFactory2 *pDXGIFactory;

        av_log(ctx, AV_LOG_VERBOSE, "Selecting d3d11va adapter %d\n", adapter);
        hr = mCreateDXGIFactory(&IID_IDXGIFactory2, (void **)&pDXGIFactory);
        if (SUCCEEDED(hr)) {
            if (FAILED(IDXGIFactory2_EnumAdapters(pDXGIFactory, adapter, &pAdapter)))
                pAdapter = NULL;
            IDXGIFactory2_Release(pDXGIFactory);
        }
    }

    if (pAdapter) {
        DXGI_ADAPTER_DESC desc;
        hr = IDXGIAdapter2_GetDesc(pAdapter, &desc);
        if (!FAILED(hr)) {
            av_log(ctx, AV_LOG_INFO, "Using device %04x:%04x (%ls).\n",
                   desc.VendorId, desc.DeviceId, desc.Description);
        }
    }

    hr = mD3D11CreateDevice(pAdapter, pAdapter ? D3D_DRIVER_TYPE_UNKNOWN : D3D_DRIVER_TYPE_HARDWARE, NULL, creationFlags, NULL, 0,
                   D3D11_SDK_VERSION, &device_hwctx->device, NULL, NULL);
    if (pAdapter)
        IDXGIAdapter_Release(pAdapter);
    if (FAILED(hr)) {
        av_log(ctx, AV_LOG_ERROR, "Failed to create Direct3D device (%lx)\n", (long)hr);
        return AVERROR_UNKNOWN;
    }

    hr = ID3D11Device_QueryInterface(device_hwctx->device, &IID_ID3D10Multithread, (void **)&pMultithread);
    if (SUCCEEDED(hr)) {
        ID3D10Multithread_SetMultithreadProtected(pMultithread, TRUE);
        ID3D10Multithread_Release(pMultithread);
    }

#if !HAVE_UWP && HAVE_DXGIDEBUG_H
    if (is_debug) {
        HANDLE dxgidebug_dll = LoadLibrary("dxgidebug.dll");
        if (dxgidebug_dll) {
            HRESULT (WINAPI  * pf_DXGIGetDebugInterface)(const GUID *riid, void **ppDebug)
                = (void *)GetProcAddress(dxgidebug_dll, "DXGIGetDebugInterface");
            if (pf_DXGIGetDebugInterface) {
                IDXGIDebug *dxgi_debug = NULL;
                hr = pf_DXGIGetDebugInterface(&IID_IDXGIDebug, (void**)&dxgi_debug);
                if (SUCCEEDED(hr) && dxgi_debug) {
                    IDXGIDebug_ReportLiveObjects(dxgi_debug, DXGI_DEBUG_ALL, DXGI_DEBUG_RLO_ALL);
                    av_log(ctx, AV_LOG_INFO, "Enabled dxgi debugging.\n");
                } else {
                    av_log(ctx, AV_LOG_WARNING, "Failed enabling dxgi debugging.\n");
                }
            } else {
                av_log(ctx, AV_LOG_WARNING, "Failed getting dxgi debug interface.\n");
            }
        } else {
            av_log(ctx, AV_LOG_WARNING, "Failed loading dxgi debug library.\n");
        }
    }
#endif

    if (av_dict_get(opts, "SHADER", NULL, 0))
        device_hwctx->BindFlags |= D3D11_BIND_SHADER_RESOURCE;

    if (av_dict_get(opts, "UAV", NULL, 0))
        device_hwctx->BindFlags |= D3D11_BIND_UNORDERED_ACCESS;

    if (av_dict_get(opts, "RTV", NULL, 0))
        device_hwctx->BindFlags |= D3D11_BIND_RENDER_TARGET;

    if (av_dict_get(opts, "SHARED", NULL, 0))
        device_hwctx->MiscFlags |= D3D11_RESOURCE_MISC_SHARED;

    return 0;
}

static int d3d11va_device_find_adapter_by_luid(AVHWDeviceContext *ctx,
                                               const LUID *luid)
{
    HRESULT hr;
    IDXGIAdapter *adapter = NULL;
    IDXGIFactory2 *factory;
    int adapter_id = 0;
    int ret = -1;

    hr = mCreateDXGIFactory(&IID_IDXGIFactory2, (void **)&factory);
    if (FAILED(hr)) {
        av_log(ctx, AV_LOG_ERROR, "CreateDXGIFactory returned error\n");
        return -1;
    }

    while (IDXGIFactory2_EnumAdapters(factory, adapter_id++, &adapter) !=
           DXGI_ERROR_NOT_FOUND) {
        DXGI_ADAPTER_DESC adapter_desc;

        hr = IDXGIAdapter2_GetDesc(adapter, &adapter_desc);
        IDXGIAdapter_Release(adapter);
        if (FAILED(hr)) {
            av_log(ctx, AV_LOG_DEBUG,
                   "IDXGIAdapter2_GetDesc returned error, try next adapter\n");
            continue;
        }

        if (adapter_desc.AdapterLuid.LowPart  == luid->LowPart &&
            adapter_desc.AdapterLuid.HighPart == luid->HighPart) {
            ret = adapter_id - 1;
            break;
        }
    }

    IDXGIFactory2_Release(factory);
    return ret;
}

#if CONFIG_CUDA
/* The LUID of the device a CUDA context runs on, asked of the context
 * itself: the device index the CUDA device context records is only filled
 * in for contexts it created, not for one the application supplied. The
 * LUID is documented as 8 bytes, matching a Windows LUID. cuDeviceGetLuid
 * is loaded optionally, so it can be absent on an older driver, which is
 * reported as ENOSYS. */
static int d3d11va_cuda_luid(AVCUDADeviceContext *cu_hw, LUID *luid)
{
    CudaFunctions *cu = cu_hw->internal->cuda_dl;
    unsigned int node_mask;
    CUcontext dummy;
    CUdevice dev;
    CUresult ret, pop;

    if (!cu->cuDeviceGetLuid)
        return AVERROR(ENOSYS);
    if (cu->cuCtxPushCurrent(cu_hw->cuda_ctx) != CUDA_SUCCESS)
        return AVERROR_EXTERNAL;
    ret = cu->cuCtxGetDevice(&dev);
    if (ret == CUDA_SUCCESS)
        ret = cu->cuDeviceGetLuid((char *)luid, &node_mask, dev);
    pop = cu->cuCtxPopCurrent(&dummy);
    return ret != CUDA_SUCCESS || pop != CUDA_SUCCESS ? AVERROR_EXTERNAL : 0;
}
#endif

static int d3d11va_device_derive(AVHWDeviceContext *ctx,
                                 AVHWDeviceContext *src_ctx,
                                 AVDictionary *opts, int flags)
{
    LUID luid;
    int adapter, ret;
    char adapter_str[16];

    if ((ret = ff_thread_once(&functions_loaded, load_functions)) != 0)
        return AVERROR_UNKNOWN;
    if (!mD3D11CreateDevice || !mCreateDXGIFactory) {
        av_log(ctx, AV_LOG_ERROR,
               "Failed to load D3D11 library or its functions\n");
        return AVERROR_UNKNOWN;
    }

    switch (src_ctx->type) {
#if CONFIG_VULKAN
    case AV_HWDEVICE_TYPE_VULKAN: {
        AVVulkanDeviceContext *src_hwctx = src_ctx->hwctx;
        PFN_vkGetPhysicalDeviceProperties2 prop_fn;
        VkPhysicalDeviceIDProperties vk_idp = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES,
        };
        VkPhysicalDeviceProperties2 vk_dev_props = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &vk_idp,
        };

        prop_fn = (PFN_vkGetPhysicalDeviceProperties2)
            src_hwctx->get_proc_addr(src_hwctx->inst,
                                     "vkGetPhysicalDeviceProperties2");
        if (!prop_fn)
            return AVERROR(ENOSYS);

        prop_fn(src_hwctx->phys_dev, &vk_dev_props);
        if (!vk_idp.deviceLUIDValid) {
            av_log(ctx, AV_LOG_VERBOSE,
                   "Source device does not expose a LUID\n");
            return AVERROR(ENOSYS);
        }

        // VK_LUID_SIZE is defined as 8, which is also the size of a LUID.
        memcpy(&luid, vk_idp.deviceLUID, sizeof(luid));
        break;
    }
#endif
#if CONFIG_CUDA
    case AV_HWDEVICE_TYPE_CUDA:
        ret = d3d11va_cuda_luid(src_ctx->hwctx, &luid);
        if (ret == AVERROR(ENOSYS)) {
            av_log(ctx, AV_LOG_VERBOSE, "cuDeviceGetLuid is unavailable\n");
            return ret;
        }
        if (ret < 0) {
            av_log(ctx, AV_LOG_ERROR, "Unable to get LUID from CUDA\n");
            return ret;
        }
        break;
#endif
    default:
        return AVERROR(ENOSYS);
    }

    adapter = d3d11va_device_find_adapter_by_luid(ctx, &luid);
    if (adapter < 0) {
        av_log(ctx, AV_LOG_ERROR, "Failed to find a d3d11va adapter matching "
               "the source device\n");
        return AVERROR(ENODEV);
    }

    snprintf(adapter_str, sizeof(adapter_str), "%d", adapter);
    return d3d11va_device_create(ctx, adapter_str, opts, flags);
}

const HWContextType ff_hwcontext_type_d3d11va = {
    .type                 = AV_HWDEVICE_TYPE_D3D11VA,
    .name                 = "D3D11VA",

    .device_hwctx_size    = sizeof(AVD3D11VADeviceContext),
    .frames_hwctx_size    = sizeof(D3D11VAFramesContext),

    .device_create        = d3d11va_device_create,
    .device_derive        = d3d11va_device_derive,
    .device_init          = d3d11va_device_init,
    .device_uninit        = d3d11va_device_uninit,
    .frames_get_constraints = d3d11va_frames_get_constraints,
    .frames_init          = d3d11va_frames_init,
    .frames_uninit        = d3d11va_frames_uninit,
    .frames_get_buffer    = d3d11va_get_buffer,
    .transfer_get_formats = d3d11va_transfer_get_formats,
    .transfer_data_to     = d3d11va_transfer_data,
    .transfer_data_from   = d3d11va_transfer_data,

    .pix_fmts             = (const enum AVPixelFormat[]){ AV_PIX_FMT_D3D11, AV_PIX_FMT_NONE },
};
