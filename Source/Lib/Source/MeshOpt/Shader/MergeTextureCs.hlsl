// Copyright (c) 2024-2026 Minmin Gong
//

static const uint32_t BlockDim = 16;

cbuffer param_cb
{
    uint32_t2 texture_size;
};

Texture2D gsplat_color_tex;

#ifdef __spirv__
[[vk::image_format("rgba8")]]
#endif
RWTexture2D<unorm float4> merged_tex;

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (any(dtid.xy >= texture_size))
    {
        return;
    }

    [branch]
    if (merged_tex[dtid.xy].a > 0.5f)
    {
        return;
    }

    merged_tex[dtid.xy] = saturate(gsplat_color_tex[dtid.xy]);
}
