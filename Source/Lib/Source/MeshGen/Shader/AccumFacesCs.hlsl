// Copyright (c) 2025 Minmin Gong
//

static const uint32_t BlockDim = 256;

cbuffer param_cb : register(b0)
{
    uint32_t num_faces;
};

Buffer<uint32_t> face_mark_buff : register(t0);

RWBuffer<uint32_t> view_counter_buff : register(u0);

[numthreads(BlockDim, 1, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (dtid.x >= num_faces)
    {
        return;
    }

    view_counter_buff[dtid.x] += face_mark_buff[dtid.x];
}
