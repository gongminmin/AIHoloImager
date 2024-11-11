// Copyright (c) 2024 Minmin Gong
//

cbuffer param_cb : register(b0)
{
    uint32_t2 texture_size;
};

RWTexture2D<uint32_t> bounding_box_tex : register(u0);

[numthreads(1, 1, 1)]
void main()
{
    static const uint32_t U2NetInputDim = 320;

    uint32_t2 bb_min = uint32_t2(bounding_box_tex[uint32_t2(0, 0)], bounding_box_tex[uint32_t2(1, 0)]);
    uint32_t2 bb_max = uint32_t2(bounding_box_tex[uint32_t2(2, 0)], bounding_box_tex[uint32_t2(3, 0)]);

    const uint32_t crop_extent = max(max(bb_max.x - bb_min.x, bb_max.y - bb_min.y) + 16, U2NetInputDim) / 2;
    const uint32_t2 crop_center = (bb_min + bb_max) / 2;
    bb_min = clamp(int32_t2(crop_center - crop_extent), int32_t2(0, 0), texture_size);
    bb_max = clamp(int32_t2(crop_center + crop_extent), int32_t2(0, 0), texture_size);

    bounding_box_tex[uint32_t2(0, 0)] = bb_min.x;
    bounding_box_tex[uint32_t2(1, 0)] = bb_min.y;
    bounding_box_tex[uint32_t2(2, 0)] = bb_max.x + 1;
    bounding_box_tex[uint32_t2(3, 0)] = bb_max.y + 1;
}
