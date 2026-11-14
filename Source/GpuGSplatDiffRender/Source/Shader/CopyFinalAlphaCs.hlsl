// Copyright (c) 2026 Minmin Gong
//

static const uint32_t BlockDim = 16;

cbuffer param_cb
{
    uint32_t2 width_height;
};

Texture2D gsplat_image;

RWTexture2D<float4> accum_color_alpha_tex;

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (any(dtid.xy >= width_height))
    {
        return;
    }

    accum_color_alpha_tex[dtid.xy] = float4(0, 0, 0, gsplat_image[dtid.xy].a);
}
