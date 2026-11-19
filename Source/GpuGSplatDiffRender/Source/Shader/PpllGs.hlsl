// Copyright (c) 2026 Minmin Gong
//

struct GsInput
{
    uint32_t point_id : TEXCOORD0;
    float4 min_max_coord : TEXCOORD1;
};

struct PsInput
{
    nointerpolation uint32_t point_id : TEXCOORD0;
    float4 pos : SV_Position;
};

[maxvertexcount(4)]
void main(point GsInput input[1],
          inout TriangleStream<PsInput> out_stream)
{
    PsInput output;
    output.point_id = input[0].point_id;

    const float2 min_coord = input[0].min_max_coord.xy;
    const float2 max_coord = input[0].min_max_coord.zw;

    output.pos = float4(min_coord.x, min_coord.y, 0.5f, 1);
    out_stream.Append(output);

    output.pos = float4(max_coord.x, min_coord.y, 0.5f, 1);
    out_stream.Append(output);

    output.pos = float4(min_coord.x, max_coord.y, 0.5f, 1);
    out_stream.Append(output);

    output.pos = float4(max_coord.x, max_coord.y, 0.5f, 1);
    out_stream.Append(output);

    out_stream.RestartStrip();
}
