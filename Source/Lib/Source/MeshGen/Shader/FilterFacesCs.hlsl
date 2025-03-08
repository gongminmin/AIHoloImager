// Copyright (c) 2025 Minmin Gong
//

static const uint32_t BlockDim = 256;

cbuffer param_cb : register(b0)
{
    uint32_t num_faces;
    uint32_t threshold;
};

Buffer<uint32_t> index_buff : register(t0);
Buffer<uint32_t> view_count_buff : register(t1);

RWBuffer<uint32_t> filtered_index_buff : register(u0);
RWBuffer<uint32_t> counter : register(u1);

[numthreads(BlockDim, 1, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (dtid.x >= num_faces)
    {
        return;
    }

    if (view_count_buff[dtid.x] > threshold)
    {
        uint32_t addr;
        InterlockedAdd(counter[0], 1, addr);

        for (uint32_t i = 0; i < 3; ++i)
        {
            filtered_index_buff[addr * 3 + i] = index_buff[dtid.x * 3 + i];
        }
    }
}
