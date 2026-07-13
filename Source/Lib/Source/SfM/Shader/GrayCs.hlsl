// Copyright (c) 2026 Minmin Gong
//

static const uint32_t BlockDim = 16;

cbuffer param_cb
{
    uint32_t2 width_height;
};

Texture2D input_tex;

#ifdef __spirv__
[[vk::image_format("r8")]]
#endif
RWTexture2D<unorm float> gray_tex;

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (any(dtid.xy >= width_height))
    {
        return;
    }

    gray_tex[dtid.xy] = dot(input_tex[dtid.xy].rgb, float3(0.299, 0.587, 0.114));
}
