// Copyright (c) 2026 Minmin Gong
//

#include "Platform.hlslh"

cbuffer param_cb
{
    uint32_t2 width_height;
};

float2 Screen2Ndc(float2 screen, uint32_t2 size)
{
    const float2 ndc = screen / size * 2 - 1;
    return float2(ndc.x, -ndc.y);
}

void main(float4 screen_pos_extents : TEXCOORD0,
          float3 color : TEXCOORD1,
          float4 conic_opacity : TEXCOORD2,
          out float2 out_screen_pos : TEXCOORD0,
          out float3 out_color : TEXCOORD1,
          out float4 out_conic_opacity : TEXCOORD2,
          out float4 out_min_max_coord : TEXCOORD3)
{
    const float2 screen_pos = screen_pos_extents.xy;
    const float2 adaptive_radius = screen_pos_extents.zw;
    float2 min_coord = Screen2Ndc(screen_pos - adaptive_radius, width_height);
    float2 max_coord = Screen2Ndc(screen_pos + adaptive_radius, width_height);

    AdjustYDir(min_coord);
    AdjustYDir(max_coord);

    out_screen_pos = screen_pos;
    out_color = color;
    out_conic_opacity = conic_opacity;
    out_min_max_coord = float4(min_coord, max_coord);
}
