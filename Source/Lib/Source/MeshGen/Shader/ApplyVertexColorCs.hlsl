// Copyright (c) 2025-2026 Minmin Gong
//

static const uint32_t BlockDim = 256;

cbuffer param_cb
{
    uint32_t num_vertices;
    float inv_scale;
    uint32_t stride;
    uint32_t pos_offset;
};

Texture3D color_vol_tex;
Buffer<float> pos_vertex_buff;

SamplerState trilinear_sampler;

RWBuffer<float> pos_color_vertex_buff;

[numthreads(BlockDim, 1, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (dtid.x >= num_vertices)
    {
        return;
    }

    const uint32_t offset = dtid.x * stride + pos_offset;
    const float3 pos_os = float3(pos_vertex_buff[offset + 0], pos_vertex_buff[offset + 1], pos_vertex_buff[offset + 2]);

    const float3 vol_coord = pos_os.zyx * inv_scale + 0.5f;
    const float3 color = saturate(color_vol_tex.SampleLevel(trilinear_sampler, vol_coord, 0).rgb);
    pos_color_vertex_buff[dtid.x * 6 + 0] = pos_os.x;
    pos_color_vertex_buff[dtid.x * 6 + 1] = pos_os.y;
    pos_color_vertex_buff[dtid.x * 6 + 2] = pos_os.z;
    pos_color_vertex_buff[dtid.x * 6 + 3] = color.r;
    pos_color_vertex_buff[dtid.x * 6 + 4] = color.g;
    pos_color_vertex_buff[dtid.x * 6 + 5] = color.b;
}
