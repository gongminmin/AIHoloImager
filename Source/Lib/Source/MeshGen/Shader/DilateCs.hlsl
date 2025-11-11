// Copyright (c) 2024 Minmin Gong
//

static const uint32_t BlockDim = 16;

cbuffer param_cb
{
    uint32_t texture_size;
};

Texture2D input_tex;

RWTexture2D<unorm float4> dilated_tex;

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (any(dtid.xy >= texture_size))
    {
        return;
    }

    float4 curr = input_tex.Load(uint32_t3(dtid.xy, 0));

    [branch]
    if (curr.a > 0)
    {
        dilated_tex[dtid.xy] = curr;
        return;
    }

    float4 sum = 0;
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            int2 coord = dtid.xy + int2(dx, dy);
            if (all(bool4(coord >= 0, coord < texture_size)))
            {
                float4 color = input_tex.Load(uint32_t3(coord, 0));
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
