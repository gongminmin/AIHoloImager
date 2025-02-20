// Copyright (c) 2025 Minmin Gong
//

Texture2D image_tex : register(t0);

SamplerState bilinear_sampler : register(s0);

float4 main(float2 texcoord : TEXCOORD0) : SV_Target0
{
    return image_tex.Sample(bilinear_sampler, texcoord);
}
