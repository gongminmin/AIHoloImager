// Copyright (c) 2026 Minmin Gong
//

static const uint32_t BlockDim = 16;

cbuffer param_cb
{
    uint32_t2 width_height;
};

Texture2D gsplat_image;

#ifdef __spirv__
[[vk::image_format("rgba8")]]
#endif
RWTexture2D<unorm float4> rendered_image;

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (any(dtid.xy >= width_height))
    {
        return;
    }

    const uint32_t2 coord = dtid.xy;

    const float4 gsplat = gsplat_image[coord];

    const float3 bg_clr = rendered_image[coord].rgb;
    rendered_image[coord] = float4(gsplat.rgb + gsplat.a * bg_clr, 1);
}
