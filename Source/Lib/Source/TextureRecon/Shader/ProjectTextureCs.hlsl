// Copyright (c) 2024 Minmin Gong
//

static const uint32_t BlockDim = 16;

cbuffer param_cb : register(b0)
{
    float4x4 camera_view_proj;
    float4x4 camera_view;
    float4x4 camera_view_it;
    float2 vp_offset;
    float2 delighted_offset;
    float2 delighted_scale;
    uint32_t texture_size;
};

Texture2D pos_tex : register(t0);
Texture2D normal_tex : register(t1);
Texture2D photo_tex : register(t2);
Texture2D<float> depth_tex : register(t3);

SamplerState point_sampler : register(s0);
SamplerState bilinear_sampler : register(s1);

RWTexture2D<unorm float4> accum_color_tex : register(u0);

float Sqr(float x)
{
    return x * x;
}

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (any(dtid.xy >= texture_size))
    {
        return;
    }

    float4 normal_ws = normal_tex.Load(uint32_t3(dtid.xy, 0));

    [branch]
    if (normal_ws.a < 0.5f)
    {
        return;
    }

    float4 pos_ws = pos_tex.Load(uint32_t3(dtid.xy, 0));

    float4 pos_ss = mul(pos_ws, camera_view_proj);
    float4 abs_pos_ss = abs(pos_ss);
    if (all(bool4(abs_pos_ss.xyz <= abs_pos_ss.w, pos_ss.z >= 0)))
    {
        pos_ss /= pos_ss.w;

        float3 normal_es = normalize(mul(normal_ws.xyz * 2 - 1, (float3x3)camera_view_it));
        if (normal_es.z > 0)
        {
            float2 coord = float2(pos_ss.x, -pos_ss.y) * 0.5f + 0.5f + vp_offset;
            if (pos_ss.z < depth_tex.SampleLevel(point_sampler, coord, 0) * 1.02f)
            {
                float4 pos_es = mul(pos_ws, camera_view);
                pos_es /= pos_es.w;
                float3 ray_es = normalize(pos_es.xyz);

                float4 color = photo_tex.SampleLevel(bilinear_sampler, (coord - delighted_offset) * delighted_scale, 0);
                float confidence = -ray_es.z * normal_es.z * color.a;
                float prev_confidence = accum_color_tex[dtid.xy].a;
                if (confidence > prev_confidence)
                {
                    accum_color_tex[dtid.xy] = float4(color.rgb, confidence);
                }
            }
        }
    }
}
