// Copyright (c) 2024 Minmin Gong
//

cbuffer param_cb : register(b0)
{
    float4x4 mvp;
    float4x4 mv;
};

void main(float3 pos : POSITION,
          float3 normal : NORMAL,
          float3 color : COLOR,
          out float3 out_color : COLOR,
          out float4 out_pos : SV_Position)
{
    float4 normal_es = mul(float4(normal, 0), mv);
    if (normal_es.z <= 0)
    {
        out_pos = float4(1, 1, 1, 0);
        out_color = float3(0, 0, 0);
    }
    else
    {
        out_pos = mul(float4(pos, 1), mvp);
        out_color = color;
    }
}
