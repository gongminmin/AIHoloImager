// Copyright (c) 2025 Minmin Gong
//

static const uint32_t BlockDim = 16;

cbuffer param_cb : register(b0)
{
    uint32_t2 dest_size;
};

Texture2D<float4> cropped_tex : register(t0);

RWTexture2D<unorm float4> delighted_tex : register(u0);

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (any(dtid.xy >= dest_size))
    {
        return;
    }

    delighted_tex[dtid.xy].a = cropped_tex[dtid.xy].a;
}
