// Copyright (c) 2026 Minmin Gong
//

struct GsInput
{
    float2 screen_pos : TEXCOORD0;
    float3 color : TEXCOORD1;
    float4 conic_opacity : TEXCOORD2;
    float4 min_max_coord : TEXCOORD3;
};

struct PsInput
{
    nointerpolation float2 screen_pos : TEXCOORD0;
    nointerpolation float3 color : TEXCOORD1;
    nointerpolation float4 conic_opacity : TEXCOORD2;
    float4 pos : SV_Position;
};

[maxvertexcount(4)]
void main(point GsInput input[1],
          inout TriangleStream<PsInput> out_stream)
{
    const float2 screen_pos = input[0].screen_pos;
    const float3 color = input[0].color;
    const float4 conic_opacity = input[0].conic_opacity;
    const float2 min_coord = input[0].min_max_coord.xy;
    const float2 max_coord = input[0].min_max_coord.zw;

    PsInput output;
    output.screen_pos = screen_pos;
    output.color = color;
    output.conic_opacity = conic_opacity;

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
