// Copyright (c) 2024-2025 Minmin Gong
//

static const uint32_t BlockDim = 16;

cbuffer param_cb
{
    uint32_t texture_size;
};

Texture2D accum_color_tex;

RWTexture2D<unorm float4> color_tex;

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
