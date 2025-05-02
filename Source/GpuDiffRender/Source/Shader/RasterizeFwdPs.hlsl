// Copyright (c) 2025 Minmin Gong
//

void main(float2 bc : TEXCOORD0, uint32_t prim_id : PRIMITIVE_ID,
          out float2 out_bc : SV_Target0, out uint32_t out_prim_id : SV_Target1)
{
    out_bc = bc;
    out_prim_id = prim_id;
}
