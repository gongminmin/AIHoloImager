// Copyright (c) 2025 Minmin Gong
//

#if ENABLE_DERIVATIVE_BC
#include "Common.hlslh"

cbuffer param_cb
{
    float4 viewport;
};

Buffer<float4> positions_buff;
Buffer<uint32_t> indices_buff;
#endif

void main(float2 bc : TEXCOORD0, uint32_t prim_id : PRIMITIVE_ID,
#if ENABLE_DERIVATIVE_BC
    float4 position : TEXCOORD1,
#endif
    out float2 out_bc : SV_Target0, out uint32_t out_prim_id : SV_Target1
#if ENABLE_DERIVATIVE_BC
    , out float4 out_derivative_bc : SV_Target2
#endif
)
{
    out_bc = bc;
    out_prim_id = prim_id;

#if ENABLE_DERIVATIVE_BC
    // Similar to Eq 2, but with respect to screen coordiante
    //
    // a0 = (x1 - x * w1) * (y2 - y * w2) - (y1 - y * w1) * (x2 - x * w2)
    // a1 = (x2 - x * w2) * (y0 - y * w0) - (y2 - y * w2) * (x0 - x * w0)
    // a2 = (x0 - x * w0) * (y1 - y * w1) - (y0 - y * w0) * (x1 - x * w1)
    // area = a0 + a1 + a2
    //
    // u = a0 / (a0 + a1 + a2)
    // v = a1 / (a0 + a1 + a2)
    //
    // da0/dx = -w1 * (y2 - y * w2) + (y1 - y * w1) * w2
    //        = -w1 * y2 + y1 * w2
    // da0/dy = (x * w1 - x1) * w2 + w1 * (x2 - x * w2)
    //        = -x1 * w2 + w1 * x2
    // da1/dx = -w2 * (y0 - y * w0) + (y2 - y * w2) * w0
    //        = -w2 * y0 + y2 * w0
    // da1/dy = (x * w2 - x2) * w0 + w2 * (x0 - x * w0)
    //        = -x2 * w0 + w2 * x0
    // da2/dx = -w0 * (y1 - y * w1) + (y0 - y * w0) * w1
    //        = -w0 * y1 + y0 * w1
    // da2/dy = (x * w0 - x0) * w1 + w0 * (x1 - x * w1)
    //        = -x0 * w1 + w0 * x1
    // darea/dx = da0/dx + da1/dx + da2/dx
    // darea/dy = da0/dy + da1/dy + da2/dy
    //
    // d{u,v}/d{x,y} = (da0/d{x,y} * area - a0 * (da0/d{x,y} + da1/d{x,y} + da2/d{x,y})) / area^2
    //               = (da0/d{x,y} - {u,v} * (da0/d{x,y} + da1/d{x,y} + da2/d{x,y})) / area

    const uint32_t fi = prim_id - 1;
    const float2 ndc_coord = position.xy / position.w;

    uint32_t vi[3];
    float4 pos[3];
    float2 to_p[3];
    for (uint32_t i = 0; i < 3; ++i)
    {
        vi[i] = indices_buff[fi * 3 + i];
        pos[i] = positions_buff[vi[i]];
        to_p[i] = pos[i].xy - ndc_coord * pos[i].w;
    }

    const float area = Cross(to_p[1], to_p[2]) + Cross(to_p[2], to_p[0]) + Cross(to_p[0], to_p[1]);
    static const float Eps = 1e-6f;
    const float inv_area = 1 / (area + (area >= 0 ? Eps : -Eps));

    const float da0_dx = -pos[1].w * pos[2].y + pos[2].w * pos[1].y;
    const float da0_dy = -pos[1].x * pos[2].w + pos[1].w * pos[2].x;
    const float da1_dx = -pos[2].w * pos[0].y + pos[2].y * pos[0].w;
    const float da1_dy = -pos[2].x * pos[0].w + pos[2].w * pos[0].x;
    const float da2_dx = -pos[0].w * pos[1].y + pos[0].y * pos[1].w;
    const float da2_dy = -pos[0].x * pos[1].w + pos[0].w * pos[1].x;

    const float2 duv_dx = DuvDxyw(out_bc, da0_dx, da1_dx, da2_dx, inv_area) / viewport.z * 2;
    const float2 duv_dy = DuvDxyw(out_bc, da0_dy, da1_dy, da2_dy, inv_area) / viewport.w * 2;

    out_derivative_bc = float4(duv_dx.x, duv_dy.x, duv_dx.y, duv_dy.y);
#endif
}
