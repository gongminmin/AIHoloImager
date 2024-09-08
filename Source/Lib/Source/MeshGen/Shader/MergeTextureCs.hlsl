// Copyright (c) 2024 Minmin Gong
//

#define BLOCK_DIM 256

cbuffer param_cb : register(b0)
{
    uint size;
};

Buffer<uint> uv_buff : register(t0);
Buffer<unorm float4> color_buffer : register(t1);

RWTexture2D<unorm float4> merged_tex : register(u0);

[numthreads(BLOCK_DIM, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (dtid.x >= size)
    {
        return;
    }

    uint uv = uv_buff[dtid.x];
    merged_tex[uint2(uv & 0xFFFF, uv >> 16)] = color_buffer[dtid.x];
}
