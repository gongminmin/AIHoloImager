// Copyright (c) 2024 Minmin Gong
//

#include "Lanczos.hlslh"

static const uint32_t BlockDim = 16;
static const uint32_t KernelRadius = 3;

cbuffer param_cb
{
    uint32_t4 src_roi;
    uint32_t2 dest_size;
    float scale;
    bool x_dir;
};

Texture2D<float> input_tex;
Texture2D<uint32_t> min_max_tex;

RWTexture2D<unorm float> output_tex;

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (any(dtid.xy >= dest_size))
    {
        return;
    }

    float min_pred;
    float pred_scale;
    if (x_dir)
    {
        min_pred = min_max_tex.Load(uint32_t3(0, 0, 0)) / 1e5f;
        const float max_pred = min_max_tex.Load(uint32_t3(1, 0, 0)) / 1e5f;
        pred_scale = 1 / (max_pred - min_pred);
    }
    else
    {
        min_pred = 0;
        pred_scale = 1;
    }

    output_tex[dtid.xy] = LanczosResample(input_tex, dtid.xy, src_roi, scale, KernelRadius, x_dir, min_pred, pred_scale);
}
