// Copyright (c) 2025 Minmin Gong
//

static const uint32_t BlockDim = 256;

cbuffer param_cb
{
    uint32_t num_features;
};

Buffer<uint32_t3> coords;

RWTexture3D<uint32_t> index_volume;

[numthreads(BlockDim, 1, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (dtid.x >= num_features)
    {
        return;
    }

    index_volume[coords[dtid.x]] = dtid.x + 1; // 0 is reserved as empty
}
