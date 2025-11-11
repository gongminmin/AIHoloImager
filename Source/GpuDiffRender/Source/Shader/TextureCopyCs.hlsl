// Copyright (c) 2025 Minmin Gong
//

static const uint32_t BlockDim = 16;

cbuffer param_cb
{
    uint32_t2 tex_size;
};

Texture2D texture;

RWTexture2D<float4> texture_f32;

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (any(dtid.xy >= tex_size))
    {
        return;
    }

    texture_f32[dtid.xy] = texture.Load(uint3(dtid.xy, 0));
}
