// Copyright (c) 2024-2025 Minmin Gong
//

cbuffer param_cb
{
    float4x4 mvp;
};

void main(float3 pos : POSITION,
          float2 texcoord0 : TEXCOORD0,
          out float4 out_pos : SV_Position)
{
    out_pos = mul(float4(pos.xyz, 1), mvp);
}
