// Copyright (c) 2024 Minmin Gong
//

#define BLOCK_DIM 16

Texture2D pos_tex : register(t0);

RWBuffer<uint> counter_buff : register(u0);
RWBuffer<uint2> uv_buff : register(u1);
RWBuffer<float> pos_buff : register(u2);

[numthreads(BLOCK_DIM, BLOCK_DIM, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    uint width;
    uint height;
    pos_tex.GetDimensions(width, height);
    [branch]
    if (any(dtid.xy > uint2(width, height)))
    {
        return;
    }

    float4 pos = pos_tex.Load(uint3(dtid.xy, 0));

    [branch]
    if (pos.a < 0.5f)
    {
        return;
    }

    uint addr;
    InterlockedAdd(counter_buff[0], 1, addr);

    uv_buff[addr] = dtid.xy;
    pos_buff[addr * 3 + 0] = pos.x;
    pos_buff[addr * 3 + 1] = pos.y;
    pos_buff[addr * 3 + 2] = pos.z;
}
