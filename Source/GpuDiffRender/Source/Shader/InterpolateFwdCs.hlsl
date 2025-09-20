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
#if ENABLE_DERIVATIVE_BC
Texture2D<float4> derivative_barycentric_tex : register(t4);
#endif

RWBuffer<float> shading : register(u0);
#if ENABLE_DERIVATIVE_BC
RWBuffer<float2> derivative_shading : register(u1);
#endif

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
#if ENABLE_DERIVATIVE_BC
    const float4 duv_dxy = derivative_barycentric_tex[dtid.xy];
#endif
    for (uint32_t i = 0; i < num_attribs; ++i)
    {
        const float3 attribs = float3(vtx_attribs_buff[in_offsets[0] + i],
                                      vtx_attribs_buff[in_offsets[1] + i],
                                      vtx_attribs_buff[in_offsets[2] + i]);

        shading[out_offset + i] = bc.x * attribs.x +
                                  bc.y * attribs.y +
                                  bc.z * attribs.z;

#if ENABLE_DERIVATIVE_BC
        // Eq 4
        // dA/d{u,v} = {A0 - A2, A1 - A2}
        const float2 da_duv = attribs.xy - attribs.z;

        // dA/d{x,y} = dA/d{u,v} * d{u,v}/d{x,y}
        const float2 da_dxy = float2(dot(da_duv, duv_dxy.xz), dot(da_duv, duv_dxy.yw));
        derivative_shading[out_offset + i] = da_dxy;
#endif
    }
}
