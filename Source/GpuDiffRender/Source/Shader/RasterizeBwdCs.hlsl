// Copyright (c) 2025 Minmin Gong
//

#include "Common.hlslh"

static const uint32_t BlockDim = 16;

cbuffer param_cb : register(b0)
{
    uint32_t2 gbuffer_size;
};

Texture2D<float2> barycentric_tex : register(t0);
Texture2D<uint32_t> prim_id_tex : register(t1);
Texture2D<float2> grad_barycentric_tex : register(t2);
Buffer<float4> positions_buff : register(t3);
Buffer<uint32_t> indices_buff : register(t4);

RWBuffer<uint32_t> grad_positions_buff : register(u0);

float2 DuvDxyw(float2 bc, float da0_di, float da1_di, float da2_di, float inv_area)
{
    return (float2(da0_di, da1_di) - bc * (da0_di + da1_di + da2_di)) * inv_area;
}

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (any(dtid.xy >= gbuffer_size))
    {
        return;
    }

    uint32_t fi = prim_id_tex[dtid.xy];
    [branch]
    if (fi == 0)
    {
        return;
    }

    float2 dl_duv = grad_barycentric_tex[dtid.xy]; // dL/d{u,v}
    [branch]
    if (all(dl_duv == 0))
    {
        return;
    }

    --fi;

    const float2 ndc_coord = WinToNdc(dtid.xy, gbuffer_size);

    uint32_t vi[3];
    float4 pos[3];
    float2 to_p[3];
    for (uint32_t i = 0; i < 3; ++i)
    {
        vi[i] = indices_buff[fi * 3 + i];
        pos[i] = positions_buff[vi[i]];
        to_p[i] = pos[i].xy - ndc_coord * pos[i].w;
    }

    // Eq 2
    // dL/dx{0,1,2} = dL/du * du/dx{0,1,2} + dL/dv * dv/dx{0,1,2}
    // dL/dy{0,1,2} = dL/du * du/dy{0,1,2} + dL/dv * dv/dy{0,1,2}
    // dL/dz{0,1,2} = 0
    // dL/dw{0,1,2} = dL/du * du/dw{0,1,2} + dL/dv * dv/dw{0,1,2}
    //
    // a0 = (x1 - x * w1) * (y2 - y * w2) - (y1 - y * w1) * (x2 - x * w2)
    // a1 = (x2 - x * w2) * (y0 - y * w0) - (y2 - y * w2) * (x0 - x * w0)
    // a2 = (x0 - x * w0) * (y1 - y * w1) - (y0 - y * w0) * (x1 - x * w1)
    // area = a0 + a1 + a2
    //
    // u = a0 / (a0 + a1 + a2)
    // v = a1 / (a0 + a1 + a2)
    //
    // d{u,v}/d{x,y,w}{0,1,2} = (da{0,1}/d{x,y,w}{0,1,2} * area - a{0,1} * (da0/d{x,y,w}{0,1,2} + da1/d{x,y,w}{0,1,2} + da2/d{x,y,w}{0,1,2})) / area^2
    //                        = (da{0,1}/d{x,y,w}{0,1,2} - {u,v} * (da0/d{x,y,w}{0,1,2} + da1/d{x,y,w}{0,1,2} + da2/d{x,y,w}{0,1,2})) / area
    //
    // da0/dx0 = 0
    // da1/dx0 = -(y2 - y * w2)
    // da2/dx0 = y1 - y * w1
    //
    // da0/dy0 = 0
    // da1/dy0 = x2 - x * w2
    // da2/dy0 = -(x1 - x * w1)
    //
    // da0/dw0 = 0
    // da1/dw0 = -x2 * y + y2 * x
    // da2/dw0 = -x * y1 + y * x1
    //
    // da0/dx1 = y2 - y * w2
    // da1/dx1 = 0
    // da2/dx1 = -(y0 - y * w0)
    //
    // da0/dy1 = -(x2 - x * w2)
    // da1/dy1 = 0
    // da2/dy1 = x0 - x * w0
    //
    // da0/dw1 = -x * y2 + y * x2
    // da1/dw1 = 0
    // da2/dw1 = -x0 * y + y0 * x
    //
    // da0/dx2 = -(y1 - y * w1)
    // da1/dx2 = y0 - y * w0
    // da2/dx2 = 0
    //
    // da0/dy2 = x1 - x * w1
    // da1/dy2 = -(x0 - x * w0)
    // da2/dy2 = 0
    //
    // da0/dw2 = -x1 * y + y1 * x
    // da1/dw2 = -x * y0 + y * x0
    // da2/dw2 = 0

    const float area = Cross(to_p[1], to_p[2]) + Cross(to_p[2], to_p[0]) + Cross(to_p[0], to_p[1]);
    static const float Eps = 1e-6f;
    const float inv_area = 1 / (area + (area >= 0 ? Eps : -Eps));

    const float2 bc = barycentric_tex[dtid.xy];

    float2 duv_dx;
    float2 duv_dy;
    float2 duv_dw;

    duv_dx = DuvDxyw(bc, 0, -to_p[2].y, to_p[1].y, inv_area);
    duv_dy = DuvDxyw(bc, 0, to_p[2].x, -to_p[1].x, inv_area);
    duv_dw = DuvDxyw(bc, 0, pos[2].y * ndc_coord.x - pos[2].x * ndc_coord.y, ndc_coord.y * pos[1].x - ndc_coord.x * pos[1].y, inv_area);
    AtomicAdd(grad_positions_buff, vi[0] * 4 + 0, dot(dl_duv, duv_dx));
    AtomicAdd(grad_positions_buff, vi[0] * 4 + 1, dot(dl_duv, duv_dy));
    AtomicAdd(grad_positions_buff, vi[0] * 4 + 3, dot(dl_duv, duv_dw));

    duv_dx = DuvDxyw(bc, to_p[2].y, 0, -to_p[0].y, inv_area);
    duv_dy = DuvDxyw(bc, -to_p[2].x, 0, to_p[0].x, inv_area);
    duv_dw = DuvDxyw(bc, ndc_coord.y * pos[2].x - ndc_coord.x * pos[2].y, 0, pos[0].y * ndc_coord.x - pos[0].x * ndc_coord.y, inv_area);
    AtomicAdd(grad_positions_buff, vi[1] * 4 + 0, dot(dl_duv, duv_dx));
    AtomicAdd(grad_positions_buff, vi[1] * 4 + 1, dot(dl_duv, duv_dy));
    AtomicAdd(grad_positions_buff, vi[1] * 4 + 3, dot(dl_duv, duv_dw));

    duv_dx = DuvDxyw(bc, -to_p[1].y, to_p[0].y, 0, inv_area);
    duv_dy = DuvDxyw(bc, to_p[1].x, -to_p[0].x, 0, inv_area);
    duv_dw = DuvDxyw(bc, pos[1].y * ndc_coord.x - pos[1].x * ndc_coord.y, ndc_coord.y * pos[0].x - ndc_coord.x * pos[0].y, 0, inv_area);
    AtomicAdd(grad_positions_buff, vi[2] * 4 + 0, dot(dl_duv, duv_dx));
    AtomicAdd(grad_positions_buff, vi[2] * 4 + 1, dot(dl_duv, duv_dy));
    AtomicAdd(grad_positions_buff, vi[2] * 4 + 3, dot(dl_duv, duv_dw));
}
