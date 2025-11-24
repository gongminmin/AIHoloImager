// Copyright (c) 2024-2025 Minmin Gong
//

static const uint32_t BlockDim = 16;

cbuffer param_cb
{
    float4x4 camera_view_proj;
    float4x4 camera_view;
    float4x4 camera_view_it;
    float2 vp_offset;
    float2 delighted_offset;
    float2 delighted_scale;
    uint32_t texture_size;
};

Texture2D pos_tex;
Texture2D normal_tex;
Texture2D photo_tex;
Texture2D<float> depth_tex;

SamplerState point_sampler;
SamplerState bilinear_sampler;

#ifdef __spirv__
[[vk::image_format("rgba8")]]
#endif
RWTexture2D<unorm float4> accum_color_tex;

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
