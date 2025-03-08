// Copyright (c) 2025 Minmin Gong
//

cbuffer param_cb : register(b0)
{
    float4x4 mvp;
};

void main(float3 pos : POSITION,
          out float4 out_pos : SV_Position)
{
    out_pos = mul(float4(pos, 1), mvp);
}
