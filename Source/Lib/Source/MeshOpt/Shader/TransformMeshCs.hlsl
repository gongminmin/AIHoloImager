// Copyright (c) 2024-2026 Minmin Gong
//

static const uint32_t BlockDim = 256;

cbuffer param_cb
{
    uint32_t num_vertices;
    uint32_t stride;
    uint32_t pos_offset;
    uint32_t normal_offset;
    float4x4 transform_mtx;
    float4x4 transform_it_mtx;
};

RWBuffer<float> vertex_buff;

[numthreads(BlockDim, 1, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID, uint32_t group_index : SV_GroupIndex)
{
    [branch]
    if (dtid.x >= num_vertices)
    {
        return;
    }

    const uint32_t vertex_offset = dtid.x * stride;

    if (pos_offset != ~0U)
    {
        float4 pos = float4(vertex_buff[vertex_offset + pos_offset + 0],
                            vertex_buff[vertex_offset + pos_offset + 1],
                            vertex_buff[vertex_offset + pos_offset + 2],
                            1);
        pos = mul(pos, transform_mtx);
        pos.xyz /= pos.w;
        vertex_buff[vertex_offset + pos_offset + 0] = pos.x;
        vertex_buff[vertex_offset + pos_offset + 1] = pos.y;
        vertex_buff[vertex_offset + pos_offset + 2] = pos.z;
    }

    if (normal_offset != ~0U)
    {
        float3 normal = float3(vertex_buff[vertex_offset + normal_offset + 0],
                               vertex_buff[vertex_offset + normal_offset + 1],
                               vertex_buff[vertex_offset + normal_offset + 2]);
        normal = mul(normal, (float3x3)transform_it_mtx);
        vertex_buff[vertex_offset + normal_offset + 0] = normal.x;
        vertex_buff[vertex_offset + normal_offset + 1] = normal.y;
        vertex_buff[vertex_offset + normal_offset + 2] = normal.z;
    }
}
