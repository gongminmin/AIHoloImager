// Copyright (c) 2024-2005 Minmin Gong
//

static const uint32_t BlockDim = 16;
static const uint32_t SsaaScale = 4;

Texture2D ssaa_tex : register(t0);

RWTexture2D<unorm float4> tex : register(u0);

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    float4 color = 0;
    for (uint32_t dy = 0; dy < SsaaScale; ++dy)
    {
        for (uint32_t dx = 0; dx < SsaaScale; ++dx)
        {
            uint32_t2 coord = dtid.xy * SsaaScale + uint32_t2(dx, dy);
            color += ssaa_tex.Load(uint32_t3(coord, 0));
        }
    }

    color /= (SsaaScale * SsaaScale);
    tex[dtid.xy] = float4(color.rgb * color.a, 1);
}
