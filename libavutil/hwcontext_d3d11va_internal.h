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

#endif /* AVUTIL_HWCONTEXT_D3D11VA_INTERNAL_H */
