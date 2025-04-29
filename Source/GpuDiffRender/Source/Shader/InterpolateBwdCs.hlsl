// Copyright (c) 2025 Minmin Gong
//

#include "Common.hlslh"

static const uint32_t BlockDim = 16;

cbuffer param_cb : register(b0)
{
    uint32_t2 gbuffer_size;
    uint32_t num_attribs;
};

Texture2D<float4> gbuffer_tex : register(t0);
Buffer<float> vtx_attribs_buff : register(t1);
Buffer<uint32_t> indices_buff : register(t2);
Buffer<float> grad_shading_buff : register(t3);

RWBuffer<uint32_t> grad_vtx_attribs : register(u0);
RWTexture2D<float4> grad_gbuffer : register(u1);

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (any(dtid.xy >= gbuffer_size))
    {
        return;
    }

    const float4 rast = gbuffer_tex[dtid.xy];
    uint32_t fi = asuint(rast.w);
    [branch]
    if (fi == 0)
    {
        return;
    }

    --fi;

    uint32_t offsets[3];
    for (uint32_t i = 0; i < 3; ++i)
    {
        offsets[i] = indices_buff[fi * 3 + i] * num_attribs;
    }

    const uint32_t pixel_offset = (dtid.y * gbuffer_size.x + dtid.x) * num_attribs;
    const float3 bc = float3(rast.xy, 1 - rast.x - rast.y);
    float2 dl_duv = 0;
    for (uint32_t i = 0; i < num_attribs; ++i)
    {
        const float dl_da = grad_shading_buff[pixel_offset + i]; // dL/dA
        if (dl_da != 0)
        {
            // dL/dA{0,1,2} = dL/dA * dA/dA{0,1,2}
            // A = u * A0 + v * A1 + (1 - u - v) * A2
            // dA/dA{0,1,2} = {u, v, 1 - u - v}
            AtomicAdd(grad_vtx_attribs, offsets[0] + i, dl_da * bc.x);
            AtomicAdd(grad_vtx_attribs, offsets[1] + i, dl_da * bc.y);
            AtomicAdd(grad_vtx_attribs, offsets[2] + i, dl_da * bc.z);

            // Eq 4
            // dL/d{u,v} = dL/dA * dA/d{u,v}
            // dA/d{u,v} = {A0 - A2, A1 - A2}
            const float3 attribs = float3(vtx_attribs_buff[offsets[0] + i],
                                          vtx_attribs_buff[offsets[1] + i],
                                          vtx_attribs_buff[offsets[2] + i]);
            dl_duv += dl_da * (attribs.xy - attribs.z);
        }
    }

    grad_gbuffer[dtid.xy] = float4(dl_duv, 0, 0);
}
