/*
 * Copyright 2019 NVIDIA Corporation
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *    http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_runtime.h>
#include "NvCodecUtils.h"

template<typename YuvUnitx2>
static __global__ void
Resize(
    cudaTextureObject_t texY,
    cudaTextureObject_t texUv,
    uint8_t *pDst,
    uint8_t *pDstUV,
    int nPitch,
    int nWidth,
    int nHeight,
    float fxScale,
    float fyScale)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x,
        iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= nWidth / 2 || iy >= nHeight / 2) {
        return;
    }

    int x = ix * 2, y = iy * 2;
    typedef decltype(YuvUnitx2::x) YuvUnit;
    const int MAX = 1 << (sizeof(YuvUnit) * 8);
    *(YuvUnitx2 *) (pDst + y * nPitch + x * sizeof(YuvUnit)) = YuvUnitx2{
        (YuvUnit) (tex2D<float>(texY, x / fxScale, y / fyScale) * MAX),
        (YuvUnit) (tex2D<float>(texY, (x + 1) / fxScale, y / fyScale) * MAX)
    };
    y++;
    *(YuvUnitx2 *) (pDst + y * nPitch + x * sizeof(YuvUnit)) = YuvUnitx2{
        (YuvUnit) (tex2D<float>(texY, x / fxScale, y / fyScale) * MAX),
        (YuvUnit) (tex2D<float>(texY, (x + 1) / fxScale, y / fyScale) * MAX)
    };
    float2 uv = tex2D<float2>(texUv, ix / fxScale, (nHeight + iy) / fyScale + 0.5f);
    *(YuvUnitx2 *) (pDstUV + iy * nPitch + ix * 2 * sizeof(YuvUnit)) =
        YuvUnitx2{(YuvUnit) (uv.x * MAX), (YuvUnit) (uv.y * MAX)};
}

template<typename YuvUnitx2>
static void
Resize(
    unsigned char *dpDst,
    unsigned char *dpDstUV,
    int nDstPitch,
    int nDstWidth,
    int nDstHeight,
    unsigned char *dpSrc,
    int nSrcPitch,
    int nSrcWidth,
    int nSrcHeight,
    cudaStream_t S)
{
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = dpSrc;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<decltype(YuvUnitx2::x)>();
    resDesc.res.pitch2D.width = nSrcWidth;
    resDesc.res.pitch2D.height = nSrcHeight;
    resDesc.res.pitch2D.pitchInBytes = nSrcPitch;

    cudaTextureDesc texDesc = {};
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;

    cudaTextureObject_t texY = 0;
    cudaCreateTextureObject(&texY, &resDesc, &texDesc, NULL);

    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<YuvUnitx2>();
    resDesc.res.pitch2D.width = nSrcWidth / 2;
    resDesc.res.pitch2D.height = nSrcHeight * 3 / 2;

    cudaTextureObject_t texUv = 0;
    cudaCreateTextureObject(&texUv, &resDesc, &texDesc, NULL);

    dim3 Dg = dim3((nDstWidth + 31) / 32, (nDstHeight + 31) / 32);
    dim3 Db = dim3(16, 16);
    Resize<YuvUnitx2> << < Dg, Db, 0, S >> > (
        texY, texUv, dpDst, dpDstUV, nDstPitch, nDstWidth, nDstHeight,
            1.0f * nDstWidth / nSrcWidth, 1.0f * nDstHeight / nSrcHeight);

    cudaDestroyTextureObject(texY);
    cudaDestroyTextureObject(texUv);
}

void
ResizeNv12(
    unsigned char *dpDstNv12,
    int nDstPitch,
    int nDstWidth,
    int nDstHeight,
    unsigned char *dpSrcNv12,
    int nSrcPitch,
    int nSrcWidth,
    int nSrcHeight,
    unsigned char *dpDstNv12UV,
    cudaStream_t S)
{
    unsigned char *dpDstUV = dpDstNv12UV ? dpDstNv12UV : dpDstNv12 + (nDstPitch * nDstHeight);
    return Resize<uchar2>(dpDstNv12,
                          dpDstUV,
                          nDstPitch,
                          nDstWidth,
                          nDstHeight,
                          dpSrcNv12,
                          nSrcPitch,
                          nSrcWidth,
                          nSrcHeight,
                          S);
}