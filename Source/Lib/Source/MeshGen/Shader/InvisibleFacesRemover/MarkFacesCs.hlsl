// Copyright (c) 2025 Minmin Gong
//

static const uint32_t BlockDim = 16;

cbuffer param_cb : register(b0)
{
    uint32_t2 width_height;
};

Texture2D<uint32_t> face_id_tex : register(t0);

RWBuffer<uint32_t> face_mark_buff : register(u0);

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (any(dtid.xy >= width_height))
    {
        return;
    }

    const uint32_t face_id = face_id_tex.Load(uint32_t3(dtid.xy, 0));
    if (face_id > 0)
    {
        face_mark_buff[face_id - 1] = 1;
    }
}
