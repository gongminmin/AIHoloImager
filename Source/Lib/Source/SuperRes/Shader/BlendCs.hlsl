// Copyright (c) 2026 Minmin Gong
//

static const uint32_t BlockDim = 16;

cbuffer param_cb
{
    uint32_t2 dest_size;
    uint32_t scale;
};

Texture3D<float4> upsampled_residual_tex;
Texture2D<float4> input_tex;

#ifdef __spirv__
[[vk::image_format("rgba8")]]
#endif
RWTexture2D<unorm float4> upsampled_net_tex;

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (any(dtid.xy >= dest_size))
    {
        return;
    }

    const uint32_t2 input_coord = dtid.xy / scale;
    const uint32_t2 coord_in_tile = dtid.xy - input_coord * scale;

    const float4 base = input_tex[input_coord];
    const float4 residual = upsampled_residual_tex[uint32_t3(input_coord, coord_in_tile.y * scale + coord_in_tile.x)];

    upsampled_net_tex[dtid.xy] = base + (residual * 2 - 1);
}
