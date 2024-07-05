void main(float2 ai_tc : TEXCOORD0,
          float2 photo_tc : TEXCOORD1,
          out float2 out_ai_tc : TEXCOORD0,
          out float2 out_photo_tc : TEXCOORD1,
          out float4 out_pos : SV_Position)
{
    out_pos = float4(ai_tc.xy * 2 - 1, 0, 1);
    out_ai_tc = float2(ai_tc.x, 1 - ai_tc.y);
    out_photo_tc = float2(photo_tc.x, 1 - photo_tc.y);
}
