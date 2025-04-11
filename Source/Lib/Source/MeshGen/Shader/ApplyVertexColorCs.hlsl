// Copyright (c) 2025 Minmin Gong
//

static const uint32_t BlockDim = 256;

cbuffer param_cb : register(b0)
{
    uint32_t num_vertices;
    float inv_scale;
};

Texture3D color_vol_tex : register(t0);
Buffer<float3> pos_vertex_buff : register(t1);

SamplerState trilinear_sampler : register(s0);

RWBuffer<float> color_vertex_buff : register(u0);

[numthreads(BlockDim, 1, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (dtid.x >= num_vertices)
    {
        return;
    }

    float3 pos_os = pos_vertex_buff[dtid.x];

    const float3 vol_coord = pos_os.zyx * inv_scale + 0.5f;
    const float3 color = saturate(color_vol_tex.SampleLevel(trilinear_sampler, vol_coord, 0).rgb);
    color_vertex_buff[dtid.x * 3 + 0] = color.r;
    color_vertex_buff[dtid.x * 3 + 1] = color.g;
    color_vertex_buff[dtid.x * 3 + 2] = color.b;
}
