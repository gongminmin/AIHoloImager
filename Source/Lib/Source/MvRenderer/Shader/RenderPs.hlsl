Texture2D diffuse_tex : register(t0);

SamplerState linear_sampler : register(s0);

float4 main(float2 texcoord0 : TEXCOORD0) : SV_Target0
{
    return float4(diffuse_tex.Sample(linear_sampler, texcoord0).rgb, 1);
}
