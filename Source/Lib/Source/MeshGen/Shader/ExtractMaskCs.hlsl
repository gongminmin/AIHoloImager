// Copyright (c) 2024 Minmin Gong
//

static const uint32_t BlockDim = 16;

cbuffer param_cb
{
    uint32_t2 texture_size;
};

Texture2D input_tex;

RWTexture2D<unorm float> mask_tex;

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (any(dtid.xy >= texture_size))
    {
        return;
    }

    mask_tex[dtid.xy] = input_tex.Load(uint32_t3(dtid.xy, 0)).a;
}
