// Copyright (c) 2025 Minmin Gong
//

cbuffer param_cb : register(b0)
{
    float4x4 rotation_mtx;
    float4 tc_bounding_box;
};

void main(uint32_t vertex_id : SV_VertexId,
          out float2 out_texcoord : TEXCOORD0,
          out float4 out_pos : SV_Position)
{
    const float2 Quad[] = {
        float2(-1, +1),
        float2(+1, +1),
        float2(-1, -1),
        float2(+1, -1),
    };

    float2 pos = Quad[vertex_id];
    out_pos = mul(float4(pos, 0, 1), rotation_mtx);
    out_texcoord = select(pos < 0, tc_bounding_box.xw, tc_bounding_box.zy);
}
