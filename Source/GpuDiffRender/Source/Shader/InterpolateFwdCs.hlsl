// Copyright (c) 2025 Minmin Gong
//

static const uint32_t BlockDim = 16;

cbuffer param_cb : register(b0)
{
    uint32_t2 gbuffer_size;
    uint32_t num_attribs;
};

Texture2D<float2> barycentric_tex : register(t0);
Texture2D<uint32_t> prim_id_tex : register(t1);
Buffer<float> vtx_attribs_buff : register(t2);
Buffer<uint32_t> indices_buff : register(t3);

RWBuffer<float> shading : register(u0);

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

    --fi;

    uint32_t in_offsets[3];
    for (uint32_t i = 0; i < 3; ++i)
    {
        in_offsets[i] = indices_buff[fi * 3 + i] * num_attribs;
    }

    const uint32_t out_offset = (dtid.y * gbuffer_size.x + dtid.x) * num_attribs;
    float3 bc;
    bc.xy = barycentric_tex[dtid.xy];
    bc.z = 1 - bc.x - bc.y;
    for (uint32_t i = 0; i < num_attribs; ++i)
    {
        shading[out_offset + i] = bc.x * vtx_attribs_buff[in_offsets[0] + i] +
                                  bc.y * vtx_attribs_buff[in_offsets[1] + i] +
                                  bc.z * vtx_attribs_buff[in_offsets[2] + i];
    }
}
