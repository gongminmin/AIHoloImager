cbuffer param_cb : register(b0)
{
    float4x4 mvp;
};

void main(float3 pos : POSITION,
          float2 texcoord0 : TEXCOORD0,
          uint vid : SV_VertexID,
          out float2 out_texcoord0 : TEXCOORD0,
          out float4 out_pos : SV_Position)
{
    out_pos = mul(float4(pos.xy, -pos.z, 1), mvp); // RH to LH
    out_texcoord0 = float2(texcoord0.x, 1 - texcoord0.y);
}
