// Copyright (c) 2025 Minmin Gong
//

struct GsInput
{
    float4 pos : POSITION;
};

struct PsInput
{
    float2 bc : TEXCOORD0;
    nointerpolation uint prim_id : PRIMITIVE_ID;
#if ENABLE_DERIVATIVE_BC
    float4 position : TEXCOORD1;
#endif
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
        output.bc = Barycentric[i];
#if ENABLE_DERIVATIVE_BC
        output.position = input[i].pos;
#endif
        out_stream.Append(output);
    }
    out_stream.RestartStrip();
}
