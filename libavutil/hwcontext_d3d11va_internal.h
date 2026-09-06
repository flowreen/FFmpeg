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

#ifndef AVUTIL_HWCONTEXT_D3D11VA_INTERNAL_H
#define AVUTIL_HWCONTEXT_D3D11VA_INTERNAL_H

#include <d3d11.h>

#include "pixfmt.h"

/**
 * A plane bridge for sharing two-plane textures with another API. Such a
 * texture cannot be shared whole: its import is reported as dedicated-only,
 * a dedicated import always maps the start of the resource, and no API
 * reports where the second plane begins. What can be shared is one texture
 * per plane, so the bridge keeps a shared single-plane texture for each
 * plane and moves the data between them and a staging texture in the
 * two-plane format. D3D11's copy functions cannot address the planes of a
 * planar texture either, so that move is a compute shader reading and
 * writing through views that each select one plane.
 */
typedef struct FFD3D11PlaneBridge {
    void                      *compiler;   /* d3dcompiler_47.dll */
    ID3D11ComputeShader       *cs[2];      /* one and two channel copies */
    ID3D11Texture2D           *staging;    /* in the two-plane format */
    ID3D11ShaderResourceView  *staging_srv[2];
    ID3D11UnorderedAccessView *staging_uav[2]; /* missing without UAV support */
    ID3D11Texture2D           *planes[2];  /* shared, NT handle if possible */
    ID3D11ShaderResourceView  *plane_srv[2];
    ID3D11UnorderedAccessView *plane_uav[2];
    int                        width;
    int                        height;
    int                        plane_w[2]; /* in texels of the plane format */
    int                        plane_h[2];
} FFD3D11PlaneBridge;

/**
 * The error a failed D3D11 call means for sharing: ENOMEM for running out
 * of memory, which a later attempt may not, and ENOSYS for everything else.
 */
int ff_d3d11va_hr_err(HRESULT hr);

/**
 * The DXGI format a D3D11VA frames context uses for a sw format. Returns
 * ENOSYS for formats it does not support.
 */
int ff_d3d11va_texture_format(enum AVPixelFormat sw_format,
                              DXGI_FORMAT *format);

/**
 * Create a texture for another API to import: shared through an NT handle,
 * or a KMT handle on a D3D11.0 runtime, which rejects NT handle sharing.
 * Running out of memory is reported as ENOMEM before any fallback is tried,
 * everything else as ENOSYS. On failure *tex is left NULL.
 */
int ff_d3d11va_shared_texture_create(ID3D11Device *dev, int width, int height,
                                     DXGI_FORMAT format, UINT bind_flags,
                                     ID3D11Texture2D **tex, void *log_ctx);

/**
 * The DXGI format a two-plane sw format is represented by, and the single
 * plane formats of its planes. Returns ENOSYS for everything else.
 */
int ff_d3d11va_bridge_formats(enum AVPixelFormat sw, DXGI_FORMAT *tex,
                              DXGI_FORMAT plane[2]);

/**
 * Create a bridge for frames of the given size and sw format. On failure
 * (including ENOSYS for an unhandled format) *bridge is left NULL.
 */
int ff_d3d11va_bridge_create(FFD3D11PlaneBridge **bridge, ID3D11Device *dev,
                             int width, int height,
                             enum AVPixelFormat sw_format, void *log_ctx);

void ff_d3d11va_bridge_free(FFD3D11PlaneBridge **bridge);

/**
 * Move the frame-sized region of subresource index of tex into the plane
 * textures (to_planes) or assemble the plane textures into it. Downloads
 * from the plane textures require staging_uav, which the caller has to
 * check. Width and height must be even, as the callers check: copies of
 * video formats only accept aligned regions. The device lock must be held;
 * the context's compute state is saved and restored.
 */
void ff_d3d11va_bridge_run(FFD3D11PlaneBridge *b, ID3D11DeviceContext *ctx,
                           ID3D11Resource *tex, unsigned index, int to_planes);

#endif /* AVUTIL_HWCONTEXT_D3D11VA_INTERNAL_H */
