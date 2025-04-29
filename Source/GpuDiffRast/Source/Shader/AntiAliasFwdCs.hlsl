// Copyright (c) 2025 Minmin Gong
//

#include "Common.hlslh"

static const uint32_t BlockDim = 16;

cbuffer param_cb : register(b0)
{
    uint32_t2 gbuffer_size;
    uint32_t num_attribs;
};

Buffer<float> shading_buff : register(t0);
Texture2D<float4> gbuffer_tex : register(t1);
Buffer<float4> positions_buff : register(t2);
Buffer<uint32_t> indices_buff : register(t3);
Buffer<uint32_t> opposite_vertices_buff : register(t4);

RWBuffer<uint32_t> anti_aliased : register(u0);
RWBuffer<uint32_t> silhouette_counter : register(u1);
RWBuffer<uint32_t> silhouette_info : register(u2);

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID, uint32_t group_index : SV_GroupIndex)
{
    [branch]
    if (any(dtid.xy >= gbuffer_size))
    {
        return;
    }

    // Step 1: Is this pixel near an edge?

    const uint32_t2 center_coord = dtid.xy;
    const uint32_t fi0 = asuint(gbuffer_tex[center_coord].w);

    const uint32_t2 right_coord = uint32_t2(min(dtid.x + 1, gbuffer_size.x - 1), dtid.y);
    const uint32_t fi1 = asuint(gbuffer_tex[right_coord].w);

    const uint32_t2 down_coord = uint32_t2(dtid.x, min(dtid.y + 1, gbuffer_size.y - 1));
    const uint32_t fi2 = asuint(gbuffer_tex[down_coord].w);

    bool is_downs[2];
    uint32_t num_candidates = 0;
    if (fi1 != fi0)
    {
        is_downs[num_candidates] = false;
        ++num_candidates;
    }
    if (fi2 != fi0)
    {
        is_downs[num_candidates] = true;
        ++num_candidates;
    }

    [branch]
    if (num_candidates == 0)
    {
        return;
    }

    SilhouetteInfo silhouette_pixels[2];
    uint32_t num_silhouette_pixels = 0;
    for (uint32_t i = 0; i < num_candidates; ++i)
    {
        uint32_t2 pixel_coord = dtid.xy;
        const bool is_down = is_downs[i];

        const uint32_t2 pixel0 = pixel_coord;
        const uint32_t2 pixel1 = pixel_coord + (is_down ? uint32_t2(0, 1) : uint32_t2(1, 0));

        const float2 zt0 = gbuffer_tex[pixel0].zw;
        const float2 zt1 = gbuffer_tex[pixel1].zw;
        const uint32_t fi0 = asuint(zt0.y);
        const uint32_t fi1 = asuint(zt1.y);

        uint32_t fi;
        if (fi0 > 0)
        {
            if (fi1 > 0)
            {
                fi = zt0.x < zt1.x ? fi0 : fi1;
            }
            else
            {
                fi = fi0;
            }
        }
        else
        {
            fi = fi1;
        }
        [branch]
        if (fi == 0)
        {
            continue;
        }

        const bool is_face1 = (fi == fi1);
        if (is_face1)
        {
            pixel_coord = pixel1;
        }

        --fi;

        float2 half_size = gbuffer_size / 2.0f;
        const float2 ndc_coord = WinToNdc(pixel_coord, gbuffer_size);

        // Step 2: Which edge is it from?

        float2 to_p[3];
        for (uint32_t i = 0; i < 3; ++i)
        {
            const uint32_t vi = indices_buff[fi * 3 + i];
            const float4 p = positions_buff[vi];
            to_p[i] = (p.xy / p.w - ndc_coord) * half_size;
            if (is_down)
            {
                Swap(to_p[i].x, to_p[i].y);
            }
        }

        const float2 opposite_edges[] = {
            to_p[2] - to_p[1],
            to_p[0] - to_p[2],
            to_p[1] - to_p[0],
        };

        uint32_t available_edges[3];
        uint32_t num_available_edges = 0;
        for (uint32_t i = 0; i < 3; ++i)
        {
            if ((to_p[(i + 1) % 3].y * to_p[(i + 2) % 3].y < 0) && (abs(opposite_edges[i].y) >= abs(opposite_edges[i].x)))
            {
                available_edges[num_available_edges] = i;
                ++num_available_edges;
            }
        }
        [branch]
        if (num_available_edges == 0)
        {
            continue;
        }

        const int sign = is_face1 ? -1 : 1;
        const float sub_areas[] = {
            Cross(to_p[1], to_p[2]),
            Cross(to_p[2], to_p[0]),
            Cross(to_p[0], to_p[1]),
        };

        uint32_t pick_edge = available_edges[0];
        float pick_dist = sign * sub_areas[pick_edge] / opposite_edges[pick_edge].y;
        for (uint32_t i = 1; i < num_available_edges; ++i)
        {
            const uint32_t test_edge = available_edges[i];
            const float test_dist = sign * sub_areas[test_edge] / opposite_edges[test_edge].y;
            if (test_dist > pick_dist)
            {
                pick_edge = test_edge;
                pick_dist = test_dist;
            }
        }

        // Step 3: Is this edge a silhouette edge?

        bool is_silhouette;
        {
            const uint32_t op = opposite_vertices_buff[fi * 3 + pick_edge];
            if (op == ~0U)
            {
                is_silhouette = true;
            }
            else
            {
                const float4 opposite = positions_buff[op];
                float2 to_opposite = (opposite.xy / opposite.w - ndc_coord) * half_size;
                if (is_down)
                {
                    Swap(to_opposite.x, to_opposite.y);
                }

                const float this_area = Cross(to_p[1] - to_p[0], to_p[2] - to_p[0]);
                const float opposite_area = Cross(to_p[(pick_edge + 2) % 3] - to_opposite, to_p[(pick_edge + 1) % 3] - to_opposite);
                is_silhouette = (opposite_area * this_area < 0);
            }
        }
        [branch]
        if (!is_silhouette)
        {
            continue;
        }

        // Step 4: What's the alpha factor from the edge?

        const float alpha = sign * (0.5f - pick_dist);
        if ((alpha >= -0.5f) && (alpha <= 0.5f) && (alpha != 0))
        {
            // We have a winner. Accumulate the AA and store its info for backward usage.

            silhouette_pixels[num_silhouette_pixels].coord = dtid.xy;
            silhouette_pixels[num_silhouette_pixels].is_down = is_down;
            silhouette_pixels[num_silhouette_pixels].is_face1 = is_face1;
            silhouette_pixels[num_silhouette_pixels].edge = pick_edge;
            silhouette_pixels[num_silhouette_pixels].alpha = alpha;

            const uint32_t pixel0_offset = (pixel0.y * gbuffer_size.x + pixel0.x) * num_attribs;
            const uint32_t pixel1_offset = (pixel1.y * gbuffer_size.x + pixel1.x) * num_attribs;
            const uint32_t output_offset = alpha > 0 ? pixel0_offset : pixel1_offset;
            for (uint32_t ai = 0; ai < num_attribs; ++ai)
            {
                AtomicAdd(anti_aliased, output_offset + ai, alpha * (shading_buff[pixel1_offset + ai] - shading_buff[pixel0_offset + ai]));
            }

            ++num_silhouette_pixels;
        }
    }

    uint32_t base;
    InterlockedAdd(silhouette_counter[0], num_silhouette_pixels, base);

    for (uint32_t i = 0; i < num_silhouette_pixels; ++i)
    {
        EncodeSilhouetteInfo(silhouette_info, base + i, silhouette_pixels[i]);
    }
}
