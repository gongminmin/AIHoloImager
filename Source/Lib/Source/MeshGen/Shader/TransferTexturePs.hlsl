Texture2D photo_tex : register(t0);

SamplerState bilinear_sampler : register(s0);

void main(float3 pos_os : TEXCOORD0,
    float2 photo_tc : TEXCOORD1,
    out float4 color_rt : SV_Target0,
    out float4 pos_rt : SV_Target1)
{
    const float4 empty_color = float4(0xFF / 255.0f, 0x7F / 255.0f, 0x27 / 255.0f, 1);

    color_rt = photo_tex.Sample(bilinear_sampler, photo_tc);
    pos_rt = 0;

    float3 diff = abs((color_rt - empty_color).rgb);
    if ((diff.r < 2 / 255.0f) && (diff.g < 20 / 255.0f) && (diff.b < 15 / 255.0f))
    {
        pos_rt = float4(pos_os, 1);
    }
}
