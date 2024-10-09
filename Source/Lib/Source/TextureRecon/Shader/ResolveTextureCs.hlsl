// Copyright (c) 2024 Minmin Gong
//

#define BLOCK_DIM 16

cbuffer param_cb : register(b0)
{
    float4x4 inv_model;
    uint texture_size;
};

Texture2D accum_color_tex : register(t0);
Texture2D pos_tex : register(t1);

RWTexture2D<unorm float4> color_tex : register(u0);
RWBuffer<uint> counter_buff : register(u1);
RWBuffer<uint> uv_buff : register(u2);
RWStructuredBuffer<float3> pos_buff : register(u3);

[numthreads(BLOCK_DIM, BLOCK_DIM, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (any(dtid.xy >= texture_size))
    {
        return;
    }

    float4 color = accum_color_tex.Load(uint3(dtid.xy, 0));
    if (color.a < 0.1f)
    {
        color = 0;

        float4 pos_ws = pos_tex.Load(uint3(dtid.xy, 0));

        [branch]
        if (pos_ws.a > 0.5f)
        {
            uint addr;
            InterlockedAdd(counter_buff[0], 1, addr);

            float4 pos_os = mul(pos_ws, inv_model);
            pos_os /= pos_os.w;

            uv_buff[addr] = (dtid.y << 16) | dtid.x;
            pos_buff[addr] = pos_os.xyz;
        }
    }
    else
    {
        color.a = 1;
    }

    color_tex[dtid.xy] = color;
}
