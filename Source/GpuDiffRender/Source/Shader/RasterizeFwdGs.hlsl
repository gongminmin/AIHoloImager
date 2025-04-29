// Copyright (c) 2025 Minmin Gong
//

cbuffer param_cb : register(b0)
{
    float2 vp_scale;
};

struct GsInput
{
    float4 pos : POSITION;
};

struct PsInput
{
    float4 bc_zw : TEXCOORD0;
    nointerpolation uint prim_id : PRIMITIVE_ID;
    float4 pos : SV_Position;
};

[maxvertexcount(3)]
void main(triangle GsInput input[3],
          uint32_t prim_id : SV_PrimitiveID,
          inout TriangleStream<PsInput> out_stream)
{
    static const float2 Barycentric[] = {float2(1, 0), float2(0, 1), float2(0, 0)};

    PsInput output;
    output.prim_id = prim_id + 1;
    for (uint32_t i = 0; i < 3; ++i)
    {
        output.pos = input[i].pos;
        output.bc_zw = float4(Barycentric[i], input[i].pos.zw);
        out_stream.Append(output);
    }
    out_stream.RestartStrip();
}
