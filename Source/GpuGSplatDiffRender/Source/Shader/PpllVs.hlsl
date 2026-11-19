// Copyright (c) 2026 Minmin Gong
//

#include "GSplatUtils.hlslh"
#include "Platform.hlslh"

#ifdef __spirv__
[[vk::binding(0)]]
#endif
cbuffer param_cb
{
    float4 viewport;
};

void main(float4 screen_pos_extents : TEXCOORD0,
          uint32_t point_id : SV_VertexID,
          out uint32_t out_point_id : TEXCOORD0,
          out float4 out_min_max_coord : TEXCOORD1)
{
    const float2 screen_pos = screen_pos_extents.xy;
    const float2 adaptive_radius = screen_pos_extents.zw;
    float2 min_coord = Screen2Ndc(screen_pos - adaptive_radius, viewport.zw);
    float2 max_coord = Screen2Ndc(screen_pos + adaptive_radius, viewport.zw);

    AdjustYDir(min_coord);
    AdjustYDir(max_coord);

    out_point_id = point_id;
    out_min_max_coord = float4(min_coord, max_coord);
}
