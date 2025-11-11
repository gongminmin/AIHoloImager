// Copyright (c) 2025 Minmin Gong
//

Texture2D image_tex;

SamplerState bilinear_sampler;

float4 main(float2 texcoord : TEXCOORD0) : SV_Target0
{
    const float4 color = image_tex.Sample(bilinear_sampler, texcoord);
    return float4(color.rgb * color.a, 1);
}
