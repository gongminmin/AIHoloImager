// Copyright (c) 2025 Minmin Gong
//

#include "Common.hlslh"

static const uint32_t BlockDim = 32;

cbuffer param_cb
{
    uint32_t bwd_block_dim;
};

Buffer<uint32_t> silhouette_counter;

RWBuffer<uint32_t> indirect_args;

[numthreads(BlockDim, 1, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (any(dtid.x >= 3))
    {
        return;
    }

    if (dtid.x == 0)
    {
        indirect_args[0] = DivUp(silhouette_counter[0], bwd_block_dim);
    }
    else
    {
        indirect_args[dtid.x] = 1;
    }
}
