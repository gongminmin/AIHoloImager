// Copyright (c) 2025 Minmin Gong
//

float4 main(float2 bc : TEXCOORD0, uint32_t prim_id : PRIMITIVE_ID) : SV_Target0
{
    return float4(bc, 0, asfloat(prim_id));
}
