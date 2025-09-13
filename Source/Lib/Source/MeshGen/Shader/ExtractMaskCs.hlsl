// Copyright (c) 2024 Minmin Gong
//

static const uint32_t BlockDim = 16;

cbuffer param_cb : register(b0)
{
    uint32_t2 texture_size;
};

Texture2D input_tex : register(t0);

RWTexture2D<unorm float> mask_tex : register(u0);

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
