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
Texture2D<uint32_t> prim_id_tex : register(t1);
Texture2D<float> depth_tex : register(t2);
Buffer<float4> positions_buff : register(t3);
Buffer<uint32_t> indices_buff : register(t4);
Buffer<uint32_t> opposite_vertices_buff : register(t5);

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
    const uint32_t center_fi = prim_id_tex[center_coord];

    uint32_t2 right_coord;
    uint32_t2 up_coord;
    RightUpCoord(center_coord, gbuffer_size, right_coord, up_coord);

    SilhouetteInfo silhouette_pixels[2];
    uint32_t num_silhouette_pixels = 0;
    for (uint32_t i = 0; i < 2; ++i)
    {
        const bool is_up = (i == 1);

        const uint32_t2 pixel0 = center_coord;
        const uint32_t2 pixel1 = is_up ? up_coord : right_coord;

        const uint32_t other_fi = prim_id_tex[pixel1];
        [branch]
        if (other_fi == center_fi)
        {
            continue;
        }

        const float z0 = depth_tex[pixel0];
        const float z1 = depth_tex[pixel1];
        const uint32_t fi0 = center_fi;
        const uint32_t fi1 = other_fi;

        uint32_t fi;
        if (fi0 > 0)
        {
            if (fi1 > 0)
            {
                fi = z0 < z1 ? fi0 : fi1;
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
        const uint32_t2 pixel_coord = is_face1 ? pixel1 : pixel0;

        --fi;

        const float2 half_size = gbuffer_size / 2.0f;
        const float2 ndc_coord = WinToNdc(pixel_coord, gbuffer_size);

        // Step 2: Which edge is it from?

        float2 to_p[3];
        for (uint32_t i = 0; i < 3; ++i)
        {
            const uint32_t vi = indices_buff[fi * 3 + i];
            const float4 p = positions_buff[vi];
            to_p[i] = (p.xy / p.w - ndc_coord) * half_size;
            if (is_up)
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
                if (is_up)
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

            silhouette_pixels[num_silhouette_pixels].coord = pixel0;
            silhouette_pixels[num_silhouette_pixels].is_up = is_up;
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
