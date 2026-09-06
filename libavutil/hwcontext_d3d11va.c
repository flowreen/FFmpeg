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
#include "hwcontext_cuda_internal.h"
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
} D3D11VAFramesContext;

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

    fmts = av_malloc_array(3, sizeof(*fmts));
    if (!fmts)
        return AVERROR(ENOMEM);

    // Don't signal support for opaque formats. Actual access would fail.
    if (s->format != DXGI_FORMAT_420_OPAQUE) {
        fmts[n++] = ctx->sw_format;
#if CONFIG_VULKAN
        fmts[n++] = AV_PIX_FMT_VULKAN;
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
