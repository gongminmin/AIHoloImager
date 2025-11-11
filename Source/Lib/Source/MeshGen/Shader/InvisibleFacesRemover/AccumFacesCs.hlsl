// Copyright (c) 2025 Minmin Gong
//

static const uint32_t BlockDim = 256;

cbuffer param_cb
{
    uint32_t num_faces;
};

Buffer<uint32_t> face_mark_buff;

RWBuffer<uint32_t> view_counter_buff;

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
