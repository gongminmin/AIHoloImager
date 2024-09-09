// Copyright (c) 2024 Minmin Gong
//

void main(float3 pos_ws : TEXCOORD0,
    float3 normal_ws : TEXCOORD1,
    out float4 pos_rt : SV_Target0,
    out float4 normal_rt : SV_Target1)
{
    pos_rt = float4(pos_ws, 1);
    normal_rt = float4(normalize(normal_ws) * 0.5f + 0.5f, 1);
}
