// Copyright (c) 2024 Minmin Gong
//

#include "Util.hlslh"

cbuffer param_cb : register(b0)
{
    uint texture_size;
};

Texture2D photo_tex : register(t0);

SamplerState bilinear_sampler : register(s0);

RWBuffer<uint> counter_buff : register(u0);
RWBuffer<uint> uv_buff : register(u1);
RWBuffer<float> pos_buff : register(u2);

float4 main(float3 pos_os : TEXCOORD0, float4 tc : TEXCOORD1) : SV_Target0
{
    float4 color = photo_tex.Sample(bilinear_sampler, tc.xy);
    if (IsEmpty(color.rgb))
    {
        uint addr;
        InterlockedAdd(counter_buff[0], 1, addr);

        uint2 uv = uint2(tc.zw * texture_size);
        uv_buff[addr] = (uv.y << 16) | uv.x;
        pos_buff[addr * 3 + 0] = pos_os.x;
        pos_buff[addr * 3 + 1] = pos_os.y;
        pos_buff[addr * 3 + 2] = pos_os.z;
    }

    return color;
}
