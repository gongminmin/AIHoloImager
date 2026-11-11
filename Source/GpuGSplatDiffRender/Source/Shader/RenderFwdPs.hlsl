// Copyright (c) 2026 Minmin Gong
//

static const float AlphaThreshold = 1 / 255.0f;

float4 main(float2 screen_pos : TEXCOORD0,
            float3 color : TEXCOORD1,
            float4 conic_opacity : TEXCOORD2,
            float4 pos : SV_Position) : SV_Target0
{
    // Resample using conic matrix (cf. "Surface Splatting" by Zwicker et al., 2001)
    const uint32_t2 coord = uint32_t2(pos.xy);
    const float2 d = coord - screen_pos;
    const float power = -0.5f * (conic_opacity.x * d.x * d.x + conic_opacity.z * d.y * d.y) - conic_opacity.y * d.x * d.y;
    clip(-power);

    // Eq 2
    const float gaussian_alpha = min(0.99f, conic_opacity.w * exp(power));
    clip(gaussian_alpha - AlphaThreshold);

    // Eq 3
    // new_color = color * gaussian_alpha * curr_transparency + curr_color
    // new_transparency = curr_transparency * (1 - gaussian_alpha)
    // By configuring src_color_blend to DstAlpha, dst_color_blend to One, src_alpha_blend to Zero, dst_alpha_blend to InvSrcAlpha, the blending result is
    // exactly the Eq 3.
    return float4(color * gaussian_alpha, gaussian_alpha);
}
