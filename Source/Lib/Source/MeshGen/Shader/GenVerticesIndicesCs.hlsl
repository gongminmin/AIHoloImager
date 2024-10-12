// Copyright (c) 2024 Minmin Gong
//

#include "MarchingCubesUtil.hlslh"

#define BLOCK_DIM 256

cbuffer param_cb : register(b0)
{
    uint size;
    uint num_non_empty_cubes;
    float isovalue;
};

Buffer<uint> edge_table : register(t0);
Buffer<uint> triangle_table : register(t1);
Buffer<float4> scalar_deformation : register(t2);
Buffer<uint> non_empty_cube_ids : register(t3);
Buffer<uint> non_empty_cube_indices : register(t4);
Buffer<uint> cube_offsets : register(t5);
Buffer<uint2> vertex_index_offsets : register(t6);

RWStructuredBuffer<float3> mesh_vertices : register(u0);
RWBuffer<uint> mesh_indices : register(u1);

float3 InterpolateVertex(float3 p0, float3 p1, float v0, float v1, float isovalue)
{
    float3 inter_p;
    if (v1 == v0)
    {
        inter_p = (p0 + p1) / 2;
    }
    else
    {
        inter_p = lerp(p0, p1, (isovalue - v0) / (v1 - v0));
    }

    return inter_p;
}

[numthreads(BLOCK_DIM, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (dtid.x >= num_non_empty_cubes)
    {
        return;
    }

    const uint OwnedEdges[] = {0, 3, 8};

    const uint3 CoordBias[] = {
        uint3(1, 0, 0),
        uint3(0, 1, 0),
        uint3(0, 0, 1),
        uint3(0, 1, 1),
        uint3(1, 0, 1),
        uint3(1, 1, 0),
    };
    const int3 CorrespondEdges[] = {
        int3(-1, 1, 9),
        int3(2, -1, 11),
        int3(4, 7, -1),
        int3(6, -1, -1),
        int3(-1, 5, -1),
        int3(-1, -1, 10),
    };

    const uint ci = dtid.x;
    const uint3 coord = DecomposeCoord(non_empty_cube_ids[ci], size);
    const uint cube_index = non_empty_cube_indices[ci];
    const uint edges = edge_table[cube_index];

    const uint vertex_base = vertex_index_offsets[ci].x;
    uint indices[12];
    {
        uint vertex_offset = vertex_base;
        for (uint i = 0; i < sizeof(OwnedEdges) / sizeof(OwnedEdges[0]); ++i)
        {
            const uint e = OwnedEdges[i];
            if (edges & (1U << e))
            {
                indices[e] = vertex_offset;
                ++vertex_offset;
            }
        }
    }
    {
        for (uint i = 0; i < sizeof(CoordBias) / sizeof(CoordBias[0]); ++i)
        {
            const uint3 bias_coord = coord + CoordBias[i];
            const uint bias_ci = cube_offsets[(bias_coord.x * size + bias_coord.y) * size + bias_coord.z];
            if (bias_ci != ~0U)
            {
                const uint bias_cube_index = non_empty_cube_indices[bias_ci];
                const uint bias_edges = edge_table[bias_cube_index];

                uint bias_vertex_index = vertex_index_offsets[bias_ci].x;
                for (uint ei = 0; ei < sizeof(OwnedEdges) / sizeof(OwnedEdges[0]); ++ei)
                {
                    if (bias_edges & (1U << OwnedEdges[ei]))
                    {
                        const int ce = CorrespondEdges[i][ei];
                        if (ce != -1)
                        {
                            indices[ce] = bias_vertex_index;
                        }
                        ++bias_vertex_index;
                    }
                }
            }
        }
    }

    {
        uint vertex_offset = vertex_base;
        for (uint i = 0; i < sizeof(OwnedEdges) / sizeof(OwnedEdges[0]); ++i)
        {
            const uint e = OwnedEdges[i];
            if (edges & (1U << e))
            {
                uint3 beg_coord;
                uint3 end_coord;
                switch (e)
                {
                case 0:
                    beg_coord = coord;
                    end_coord = coord + uint3(1, 0, 0);
                    break;

                case 3:
                    beg_coord = coord + uint3(0, 1, 0);
                    end_coord = coord;
                    break;

                case 8:
                default:
                    beg_coord = coord;
                    end_coord = coord + uint3(0, 0, 1);
                    break;
                }

                const float4 beg_scalar_deformation = scalar_deformation[CalcOffset(beg_coord, size)];
                const float4 end_scalar_deformation = scalar_deformation[CalcOffset(end_coord, size)];
                const float3 beg_p = beg_coord + beg_scalar_deformation.yzw;
                const float3 end_p = end_coord + end_scalar_deformation.yzw;
                const float beg_scalar = beg_scalar_deformation.x;
                const float end_scalar = end_scalar_deformation.x;
                mesh_vertices[vertex_offset] = InterpolateVertex(beg_p, end_p, beg_scalar, end_scalar, isovalue);
                ++vertex_offset;
            }
        }
    }

    const uint index_base = vertex_index_offsets[ci].y;
    const uint num = triangle_table[cube_index * 16 + 0];
    for (uint m = 0; m < num; ++m)
    {
        mesh_indices[index_base + m] = indices[triangle_table[cube_index * 16 + 1 + m]];
    }
}
