Texture2D ai_tex : register(t0);
Texture2D photo_tex : register(t1);

SamplerState bilinear_sampler : register(s0);

float4 main(float2 ai_tc : TEXCOORD0, float2 photo_tc : TEXCOORD1) : SV_Target0
{
    const float4 empty_color = float4(0xFF / 255.0f, 0x7F / 255.0f, 0x27 / 255.0f, 1);

    float4 color = photo_tex.Sample(bilinear_sampler, photo_tc);
    float3 diff = abs((color - empty_color).rgb);
    if ((diff.r < 2 / 255.0f) && (diff.g < 20 / 255.0f) && (diff.b < 15 / 255.0f))
    {
        color = ai_tex.Sample(bilinear_sampler, ai_tc);
    }

    return color;
}
