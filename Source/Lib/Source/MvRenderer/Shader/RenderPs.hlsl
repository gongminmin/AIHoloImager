// Copyright (c) 2024 Minmin Gong
//

Texture2D diffuse_tex : register(t0);

SamplerState point_sampler : register(s0);

float4 main(float2 texcoord0 : TEXCOORD0) : SV_Target0
{
    return diffuse_tex.Sample(point_sampler, texcoord0);
}
