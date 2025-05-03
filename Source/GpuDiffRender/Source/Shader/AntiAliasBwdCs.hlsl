// Copyright (c) 2025 Minmin Gong
//

#include "Common.hlslh"

static const uint32_t BlockDim = 256;

cbuffer param_cb : register(b0)
{
    uint32_t2 gbuffer_size;
    uint32_t num_attribs;
};

Buffer<float> shading_buff : register(t0);
Texture2D<uint32_t> prim_id_tex : register(t1);
Buffer<float4> positions_buff : register(t2);
Buffer<uint32_t> indices_buff : register(t3);
Buffer<uint32_t> silhouette_counter : register(t4);
Buffer<uint32_t> silhouette_info : register(t5);
Buffer<float> grad_anti_aliased : register(t6);

RWBuffer<uint32_t> grad_shading : register(u0);
RWBuffer<uint32_t> grad_positions : register(u1);

[numthreads(BlockDim, 1, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID, uint32_t group_index : SV_GroupIndex)
{
    [branch]
    if (dtid.x >= silhouette_counter[0])
    {
        return;
    }

    const SilhouetteInfo silhouette_pixel = DecodeSilhouetteInfo(silhouette_info, dtid.x);

    const uint32_t2 center_coord = silhouette_pixel.coord;
    uint32_t2 right_coord;
    uint32_t2 down_coord;
    RightDownCoord(center_coord, gbuffer_size, right_coord, down_coord);

    const uint32_t2 pixel0 = center_coord;
    const uint32_t2 pixel1 = silhouette_pixel.is_down ? down_coord : right_coord;
    const uint32_t2 pixel_coord = silhouette_pixel.is_face1 ? pixel1 : pixel0;

    uint32_t fi = prim_id_tex[pixel_coord];
    [branch]
    if (fi == 0)
    {
        return;
    }

    --fi;

    float grad_factor = 0;
    {
        const uint32_t pixel0_offset = (pixel0.y * gbuffer_size.x + pixel0.x) * num_attribs;
        const uint32_t pixel1_offset = (pixel1.y * gbuffer_size.x + pixel1.x) * num_attribs;
        const uint32_t grad_offset = silhouette_pixel.alpha > 0 ? pixel0_offset : pixel1_offset;

        for (uint32_t i = 0; i < num_attribs; ++i)
        {
            const float dl_da = grad_anti_aliased[grad_offset + i]; // dL/dA
            if (dl_da != 0)
            {
                grad_factor += dl_da * (shading_buff[pixel1_offset + i] - shading_buff[pixel0_offset + i]);

                // Fig 3
                const float grad_color = dl_da * silhouette_pixel.alpha;
                AtomicAdd(grad_shading, pixel0_offset + i, -grad_color);
                AtomicAdd(grad_shading, pixel1_offset + i, +grad_color);
            }
        }
    }
    [branch]
    if (grad_factor == 0)
    {
        return;
    }

    float2 half_size = gbuffer_size / 2.0f;
    float2 ndc_coord = WinToNdc(pixel_coord, gbuffer_size) * half_size;
    if (silhouette_pixel.is_down)
    {
        Swap(half_size.x, half_size.y);
        Swap(ndc_coord.x, ndc_coord.y);
    }

    uint32_t edge_vi[2];
    float4 pos[2];
    float2 to_p[2];
    for (uint32_t i = 0; i < 2; ++i)
    {
        edge_vi[i] = indices_buff[fi * 3 + ((silhouette_pixel.edge + 1 + i) % 3)];
        pos[i] = positions_buff[edge_vi[i]];
        if (silhouette_pixel.is_down)
        {
            Swap(pos[i].x, pos[i].y);
        }

        pos[i].xy *= half_size;
        to_p[i] = pos[i].xy / pos[i].w - ndc_coord;
    }

    // dist = -cross(p0 / w0 - p, p1 / w1 - p) / (y1 / w1 - y0 / w0)
    //      = -((x0 / w0 - x) * (y1 / w1 - y) - (y0 / w0 - y) * (x1 / w1 - x)) / (y1 / w1 - y0 / w0)
    //      = (-x0 / w0 * y1 / w1 + x0 / w0 * y - x * y1 / w1 + y0 / w0 * x1 / w1 - y0 / w0 * x - y * x1 / w1)) / (y1 / w1 - y0 / w0)
    //
    // ddist/dx0 = (-y1 / w0 / w1 + y / w0) / (y1 / w1 - y0 / w0)
    // ddist/dy0 = (x1 / w0 / w1 - x / w0) / (y1 / w1 - y0 / w0) - area / w0 / (y1 / w1 - y0 / w0) ^ 2
    // ddist/dw0 = (x0 * y1 / w1 - x0 * y - y0 * x1 / w1 + y0 * x) / (y1 / w1 - y0 / w0) + area * y0 / (y1 / w1 - y0 / w0) ^ 2
    // 
    // ddist/dx1 = (y0 / w0 / w1 - y / w1) / (y1 / w1 - y0 / w0)
    // ddist/dy1 = (-x0 / w0 / w1 + x / w1) / (y1 / w1 - y0 / w0) + area / w1 / (y1 / w1 - y0 / w0) ^ 2
    // ddist/dw1 = (x0 * y1 / w0 - x * y1 - y0 * x1 / w0 + y * x1) / (y1 / w1 - y0 / w0) - area * y1 / (y1 / w1 - y0 / w0) ^ 2

    const float inv_y = 1 / (to_p[1].y - to_p[0].y);
    const float neg_dist = Cross(to_p[0], to_p[1]) * inv_y;

    float3 grad_pos[2];

    grad_pos[0].xy = half_size * float2(-to_p[1].y, to_p[1].x - neg_dist) / pos[0].w;
    grad_pos[0].z = (pos[0].x * pos[1].y - pos[0].y * pos[1].x) / pos[1].w - (pos[0].x * ndc_coord.y - pos[0].y * ndc_coord.x) + neg_dist * pos[0].y;

    grad_pos[1].xy = half_size * float2(+to_p[0].y, neg_dist - to_p[0].x) / pos[1].w;
    grad_pos[1].z = (pos[0].x * pos[1].y - pos[0].y * pos[1].x) / pos[0].w + (pos[1].x * ndc_coord.y - pos[1].y * ndc_coord.x) - neg_dist * pos[1].y;

    for (uint32_t i = 0; i < 2; ++i)
    {
        grad_pos[i] *= grad_factor * inv_y;
        if (silhouette_pixel.is_down)
        {
            Swap(grad_pos[i].x, grad_pos[i].y);
        }

        AtomicAdd(grad_positions, edge_vi[i] * 4 + 0, grad_pos[i].x);
        AtomicAdd(grad_positions, edge_vi[i] * 4 + 1, grad_pos[i].y);
        AtomicAdd(grad_positions, edge_vi[i] * 4 + 3, grad_pos[i].z);
    }
}
