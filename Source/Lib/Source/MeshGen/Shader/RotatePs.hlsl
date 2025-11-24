// Copyright (c) 2025 Minmin Gong
//

#ifdef __spirv__
[[vk::binding(1)]]
#endif
Texture2D image_tex;

#ifdef __spirv__
[[vk::binding(2)]]
#endif
SamplerState bilinear_sampler;

float4 main(float2 texcoord : TEXCOORD0) : SV_Target0
{
    const float4 color = image_tex.Sample(bilinear_sampler, texcoord);
    return float4(color.rgb * color.a, 1);
}
