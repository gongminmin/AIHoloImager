// Copyright (c) 2024 Minmin Gong
//

#define BLOCK_DIM 16

cbuffer param_cb : register(b0)
{
    uint32_t texture_size;
};

Texture2D blended_tex : register(t0);

RWTexture2D<unorm float4> dilated_tex : register(u0);

[numthreads(BLOCK_DIM, BLOCK_DIM, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (any(dtid.xy >= texture_size))
    {
        return;
    }

    float4 curr = blended_tex.Load(uint32_t3(dtid.xy, 0));

    [branch]
    if (curr.a > 0)
    {
        dilated_tex[dtid.xy] = curr;
        return;
    }

    uint32_t width;
    uint32_t height;
    blended_tex.GetDimensions(width, height);

    float4 sum = 0;
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            int2 coord = dtid.xy + int2(dx, dy);
            if (all(bool4(coord >= 0, coord < int2(width, height))))
            {
                float4 color = blended_tex.Load(uint32_t3(coord, 0));
                if (color.a > 0)
                {
                    sum += float4(color.rgb, 1);
                }
            }
        }
    }

    if (sum.a > 0)
    {
        sum.rgb /= sum.a;
    }

    dilated_tex[dtid.xy] = sum;
}
