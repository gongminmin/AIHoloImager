void main(float3 pos : POSITION,
          float2 tc : TEXCOORD0,
          out float3 out_pos_os : TEXCOORD0,
          out float4 out_pos : SV_Position)
{
    out_pos_os = pos;
    out_pos = float4(tc.xy * 2 - 1, 0, 1);
}
