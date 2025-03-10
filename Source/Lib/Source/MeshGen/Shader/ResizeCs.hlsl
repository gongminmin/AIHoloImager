// Copyright (c) 2025 Minmin Gong
//

#include "Lanczos.hlslh"

static const uint32_t BlockDim = 16;
static const uint32_t KernelRadius = 3;

cbuffer param_cb : register(b0)
{
    uint32_t4 src_roi;
    uint32_t2 dest_size;
    float scale;
    bool x_dir;
};

Texture2D<float4> input_tex : register(t0);

RWTexture2D<unorm float4> output_tex : register(u0);

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (any(dtid.xy >= dest_size))
    {
        return;
    }

    output_tex[dtid.xy] = LanczosResample(input_tex, dtid.xy, src_roi, scale, KernelRadius, x_dir, float4(0, 0, 0, 0), float4(1, 1, 1, 1));
}
