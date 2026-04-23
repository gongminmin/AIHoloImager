// Copyright (c) 2026 Minmin Gong
//

#include "GSplatUtils.hlslh"
#include "Utils.hlslh"

static const uint32_t BlockSize = ImgTileX * ImgTileY;
static const uint32_t MinWaveSize = 16;

cbuffer param_cb
{
    uint32_t2 width_height;
};

Buffer<uint32_t2> ranges_buff;
Buffer<uint32_t> point_ids_buff;
Buffer<float2> screen_pos_buff;
Buffer<float> point_colors_buff;
Buffer<float4> conic_opacity_buff;

#ifdef __spirv__
[[vk::image_format("rgba8")]]
#endif
RWTexture2D<unorm float4> rendered_image;

groupshared uint32_t group_point_id[BlockSize];
groupshared float2 group_point_screen_pos[BlockSize];
groupshared float4 group_point_conic_opacity[BlockSize];

[numthreads(ImgTileX, ImgTileY, 1)]
void main(uint32_t3 group_id : SV_GroupID, uint32_t3 dtid : SV_DispatchThreadID, uint32_t group_index : SV_GroupIndex)
{
    const uint32_t2 coord = dtid.xy;

    const uint32_t wave_size = WaveGetLaneCount();
    const uint32_t wave_index = group_index / wave_size;
    const uint32_t num_active_waves = DivUp(BlockSize, wave_size);

    const uint32_t2 range = ranges_buff[group_id.y * DivUp(width_height.x, ImgTileX) + group_id.x];
    uint32_t rest = range.y - range.x;
    const uint32_t num_rounds = DivUp(rest, BlockSize);

    const bool inside = all(coord < width_height);
    bool done = !inside;

    float transparency = 1.0f;
    float3 color = float3(0, 0, 0);

    for (uint32_t i = 0; i < num_rounds; ++i, rest -= BlockSize)
    {
        // Borrow group_point_id for calculating num_done
        uint32_t wave_done = WaveActiveSum(done ? 1 : 0);
        if (WaveIsFirstLane())
        {
            group_point_id[wave_index] = wave_done;
        }
        GroupMemoryBarrierWithGroupSync();

        if (group_index < num_active_waves)
        {
            wave_done = WaveActiveSum(group_point_id[group_index]);
        }
        GroupMemoryBarrierWithGroupSync();

        if (group_index == 0)
        {
            group_point_id[0] = wave_done;
        }
        GroupMemoryBarrierWithGroupSync();

        const uint32_t num_done = group_point_id[0];
        if (num_done == BlockSize)
        {
            break;
        }

        GroupMemoryBarrierWithGroupSync();

        const uint32_t progress = range.x + i * BlockSize + group_index;
        if (progress < range.y)
        {
            const uint32_t point_id = point_ids_buff[progress];
            group_point_id[group_index] = point_id;
            group_point_screen_pos[group_index] = screen_pos_buff[point_id];
            group_point_conic_opacity[group_index] = conic_opacity_buff[point_id];
        }
        GroupMemoryBarrierWithGroupSync();

        for (uint32_t j = 0; !done && (j < min(BlockSize, rest)); ++j)
        {
            // Resample using conic matrix (cf. "Surface Splatting" by Zwicker et al., 2001)
            const float2 d = group_point_screen_pos[j] - coord;
            const float4 conic_opacity = group_point_conic_opacity[j];
            const float power = -0.5f * (conic_opacity.x * d.x * d.x + conic_opacity.z * d.y * d.y) - conic_opacity.y * d.x * d.y;
            if (power > 0)
            {
                continue;
            }

            // Eq. 2
            const float gaussian_alpha = min(0.99f, conic_opacity.w * exp(power));
            if (gaussian_alpha < 1 / 255.0f)
            {
                continue;
            }

            const float new_transparency = transparency * (1 - gaussian_alpha);
            if (new_transparency < 1e-4f)
            {
                done = true;
                continue;
            }

            // Eq. 3
            {
                const uint32_t point_id = group_point_id[j];
                const float3 point_color = {point_colors_buff[point_id * 3 + 0], point_colors_buff[point_id * 3 + 1], point_colors_buff[point_id * 3 + 2]};
                color += point_color * gaussian_alpha * transparency;
            }

            transparency = new_transparency;
        }
    }

    if (inside)
    {
        const uint32_t2 target_coord = {coord.x, width_height.y - 1 - coord.y};
        const float3 bg_clr = rendered_image[target_coord].rgb;
        rendered_image[target_coord] = float4(color + transparency * bg_clr, 1);
    }
}
