// Copyright (c) 2024 Minmin Gong
//

static const uint32_t BlockDim = 16;

cbuffer param_cb
{
    uint32_t2 texture_size;
    uint32_t4 roi;
};

Texture2D<float> mask_tex;

#ifdef __spirv__
[[vk::image_format("rgba8")]]
#endif
RWTexture2D<unorm float4> output_tex;

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (any(dtid.xy >= texture_size))
    {
        return;
    }

    float mask = 0;
    if ((dtid.x >= roi.x) && (dtid.x < roi.z) && (dtid.y >= roi.y) && (dtid.y < roi.w))
    {
        mask = mask_tex.Load(uint32_t3(dtid.xy - roi.xy, 0));
    }

    output_tex[dtid.xy].w = mask;
}
