// Copyright (c) 2025 Minmin Gong
//

static const uint32_t BlockDim = 16;

cbuffer param_cb : register(b0)
{
    uint32_t texture_size;
};

Texture3D input_tex : register(t0);

RWTexture3D<unorm float4> dilated_tex : register(u0);

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (any(dtid >= texture_size))
    {
        return;
    }

    float4 curr = input_tex.Load(uint32_t4(dtid, 0));

    [branch]
    if (curr.a > 0)
    {
        dilated_tex[dtid] = curr;
        return;
    }

    float4 sum = 0;
    for (int dz = -1; dz <= 1; ++dz)
    {
        for (int dy = -1; dy <= 1; ++dy)
        {
            for (int dx = -1; dx <= 1; ++dx)
            {
                int3 coord = dtid + int3(dx, dy, dz);
                if (all(coord >= 0) && all(coord < texture_size))
                {
                    float4 color = input_tex.Load(uint32_t4(coord, 0));
                    if (color.a > 0)
                    {
                        sum += float4(color.rgb, 1);
                    }
                }
            }
        }
    }

    if (sum.a > 0)
    {
        sum.rgb /= sum.a;
    }

    dilated_tex[dtid] = sum;
}
