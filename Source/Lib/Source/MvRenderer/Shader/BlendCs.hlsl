// Copyright (c) 2024 Minmin Gong
//

#include "Util.hlslh"

#define BLOCK_DIM 16

cbuffer param_cb : register(b0)
{
    uint4 atlas_offset_view_size;
    uint4 rendered_diffusion_center;
    float2 diffusion_inv_size;
};

Texture2D diffusion_tex : register(t0);
Texture2D<uint> bb_tex : register(t1);

SamplerState bilinear_sampler : register(s0);

RWTexture2D<unorm float4> rendered_tex : register(u0);

[numthreads(BLOCK_DIM, BLOCK_DIM, 1)]
void main(uint3 dtid : SV_DispatchThreadID, uint group_index : SV_GroupIndex)
{
    const float ValidThreshold = 237 / 255.0f;

    float4 rendered_color = rendered_tex[dtid.xy];
    if ((rendered_color.a == 0) || IsEmpty(rendered_color.rgb))
    {
        uint2 rendered_min = uint2(bb_tex.Load(uint3(0, 0, 0)), bb_tex.Load(uint3(1, 0, 0)));
        uint2 rendered_max = uint2(bb_tex.Load(uint3(2, 0, 0)), bb_tex.Load(uint3(3, 0, 0)));
        uint2 diffusion_min = uint2(bb_tex.Load(uint3(0, 1, 0)), bb_tex.Load(uint3(1, 1, 0)));
        uint2 diffusion_max = uint2(bb_tex.Load(uint3(2, 1, 0)), bb_tex.Load(uint3(3, 1, 0)));

        float2 scale_xy = float2(diffusion_max - diffusion_min) / (rendered_max - rendered_min);
        float scale = min(scale_xy.x, scale_xy.y);

        int2 atlas_offset = atlas_offset_view_size.xy;
        int2 view_size = atlas_offset_view_size.zw;
        int2 rendered_center = rendered_diffusion_center.xy;
        int2 diffusion_center = rendered_diffusion_center.zw;

        uint2 diffusion_coord = atlas_offset + clamp(int2(round((int2(dtid.xy) - rendered_center) * scale) + diffusion_center), 0, view_size - 1);
        float3 diffusion_point_color = diffusion_tex.Load(uint3(diffusion_coord, 0)).rgb;
        if (any(diffusion_point_color < ValidThreshold))
        {
            float3 diffusion_linear_color = diffusion_tex.SampleLevel(bilinear_sampler, diffusion_coord * diffusion_inv_size, 0).rgb;
            rendered_tex[dtid.xy] = float4(min(diffusion_point_color, diffusion_linear_color), 1);
        }
    }
}
