// Copyright (c) 2024 Minmin Gong
//

#define BLOCK_DIM 16
#define MIN_WAVE_SIZE 32

Texture2D rendered_tex : register(t0);

RWTexture2D<uint32_t> bounding_box_tex : register(u0);

groupshared uint32_t2 group_wave_mins[BLOCK_DIM * BLOCK_DIM / MIN_WAVE_SIZE];
groupshared uint32_t2 group_wave_maxs[BLOCK_DIM * BLOCK_DIM / MIN_WAVE_SIZE];

[numthreads(BLOCK_DIM, BLOCK_DIM, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID, uint32_t group_index : SV_GroupIndex)
{
    uint32_t width;
    uint32_t height;
    rendered_tex.GetDimensions(width, height);

    uint32_t2 bb_min = uint32_t2(width, height);
    uint32_t2 bb_max = uint32_t2(0, 0);
    if (all(dtid.xy < uint32_t2(width, height)))
    {
        float alpha = rendered_tex.Load(uint32_t3(dtid.xy, 0)).a;
        if (alpha > 1 / 255.0f)
        {
            bb_min = dtid.xy;
            bb_max = dtid.xy;
        }
    }

    uint32_t wave_size = WaveGetLaneCount();
    uint32_t wave_index = group_index / wave_size;
    uint32_t lane_index = WaveGetLaneIndex();

    bb_min = WaveActiveMin(bb_min);
    bb_max = WaveActiveMax(bb_max);
    if (WaveIsFirstLane())
    {
        group_wave_mins[wave_index] = bb_min;
        group_wave_maxs[wave_index] = bb_max;
    }
    GroupMemoryBarrierWithGroupSync();

    uint32_t group_mem_size = BLOCK_DIM * BLOCK_DIM / wave_size;

    if (group_index < group_mem_size)
    {
        bb_min = group_wave_mins[group_index];
        bb_max = group_wave_maxs[group_index];
    }
    GroupMemoryBarrier();

    if (group_index < group_mem_size)
    {
        bb_min = WaveActiveMin(bb_min);
        bb_max = WaveActiveMax(bb_max);
    }

    if (group_index == 0)
    {
        InterlockedMin(bounding_box_tex[uint32_t2(0, 0)], bb_min.x);
        InterlockedMin(bounding_box_tex[uint32_t2(1, 0)], bb_min.y);
        InterlockedMax(bounding_box_tex[uint32_t2(2, 0)], bb_max.x);
        InterlockedMax(bounding_box_tex[uint32_t2(3, 0)], bb_max.y);
    }
}
