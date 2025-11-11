// Copyright (c) 2024-2025 Minmin Gong
//

cbuffer param_cb
{
    float4x4 model_mtx;
    float4x4 model_it_mtx;
};

void main(float3 pos : POSITION,
    float3 normal : NORMAL,
    float2 tc : TEXCOORD0,
    out float3 out_pos_ws : TEXCOORD0,
    out float3 out_normal_ws : TEXCOORD1,
    out float4 out_pos : SV_Position)
{
    out_pos = float4(tc.xy * 2 - 1, 0, 1);
    out_pos_ws = mul(float4(pos, 1), model_mtx).xyz;
    out_normal_ws = mul(normal, (float3x3)model_it_mtx);
}
