// Copyright (c) 2025 Minmin Gong
//

float4 main(float4 bc_zw : TEXCOORD0, uint prim_id : PRIMITIVE_ID) : SV_Target0
{
    return float4(bc_zw.xy, bc_zw.z / bc_zw.w, asfloat(prim_id));
}
