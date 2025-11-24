// Copyright (c) 2025 Minmin Gong
//

static const uint32_t BlockDim = 16;

cbuffer param_cb
{
    uint32_t2 dest_size;
};

Texture2D<float4> input_tex;

#ifdef __spirv__
[[vk::image_format("rgba8")]]
#endif
RWTexture2D<unorm float4> delighted_tex;

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (any(dtid.xy >= dest_size))
    {
        return;
    }

    delighted_tex[dtid.xy].a = input_tex[dtid.xy].a;
}
