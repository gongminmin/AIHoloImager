// Copyright (c) 2026 Minmin Gong
//

#include "ColorSpace.hlslh"
#include "GSplatUtils.hlslh"

static const uint32_t BlockDim = 16;

cbuffer param_cb
{
    uint32_t2 width_height;
    float3 bg_color;
    bool from_bg_color;
};

Texture2D gsplat_image;

#ifdef FLOAT3_BUFFER_TARGET
RWBuffer<float> rendered_image;
#else
#ifdef __spirv__
[[vk::image_format("rgba8")]]
#endif
RWTexture2D<unorm float4> rendered_image;
#endif

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

#ifdef FLOAT3_BUFFER_TARGET
    const uint32_t pixel_index = coord.y * width_height.x + coord.x;
    const float3 bg_clr = from_bg_color ? bg_color : ReadFloat3(rendered_image, pixel_index);
#else
    const float3 bg_clr = from_bg_color ? bg_color : SRGBToLinear(rendered_image[coord].rgb);
#endif
    const float3 color = gsplat.rgb + gsplat.a * bg_clr;
#ifdef FLOAT3_BUFFER_TARGET
    WriteFloat3(rendered_image, pixel_index, color);
#else
    rendered_image[coord] = float4(LinearToSRGB(color), 1);
#endif
}
