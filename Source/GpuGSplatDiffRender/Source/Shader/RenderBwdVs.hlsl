// Copyright (c) 2026 Minmin Gong
//

#include "GSplatUtils.hlslh"
#include "Platform.hlslh"

cbuffer param_cb
{
    float4 viewport;
};

Buffer<float4> screen_pos_extents_buff;
Buffer<float3> color_buff;
Buffer<float4> conic_opacity_buff;
Buffer<uint32_t> index_buff;
Buffer<uint32_t> num_visible_buff;

void main(uint32_t vertex_id : SV_VertexID,
          out float3 out_screen_pos_point_id : TEXCOORD0,
          out float3 out_color : TEXCOORD1,
          out float4 out_conic_opacity : TEXCOORD2,
          out float4 out_min_max_coord : TEXCOORD3)
{
    const uint32_t point_id = index_buff[num_visible_buff[0] - 1 - vertex_id]; // Revert the order
    const float4 screen_pos_extents = screen_pos_extents_buff[point_id];
    const float3 color = color_buff[point_id];
    const float4 conic_opacity = conic_opacity_buff[point_id];

    const float2 screen_pos = screen_pos_extents.xy;
    const float2 adaptive_radius = screen_pos_extents.zw;
    float2 min_coord = Screen2Ndc(screen_pos - adaptive_radius, viewport.zw);
    float2 max_coord = Screen2Ndc(screen_pos + adaptive_radius, viewport.zw);

    AdjustYDir(min_coord);
    AdjustYDir(max_coord);

    out_screen_pos_point_id = float3(screen_pos + viewport.xy, point_id != 0xFFFFFFFFU ? point_id + 0.5f : -1);
    out_color = color;
    out_conic_opacity = conic_opacity;
    out_min_max_coord = float4(min_coord, max_coord);
}
