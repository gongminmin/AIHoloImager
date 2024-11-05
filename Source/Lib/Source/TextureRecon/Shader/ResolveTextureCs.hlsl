// Copyright (c) 2024 Minmin Gong
//

static const uint32_t BlockDim = 16;

cbuffer param_cb : register(b0)
{
    uint32_t texture_size;
};

Texture2D accum_color_tex : register(t0);

RWTexture2D<unorm float4> color_tex : register(u0);

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (any(dtid.xy >= texture_size))
    {
        return;
    }

    float4 color = accum_color_tex.Load(uint32_t3(dtid.xy, 0));
    if (color.a < 0.1f)
    {
        color = 0;
    }
    else
    {
        color.a = 1;
    }

    color_tex[dtid.xy] = color;
}
