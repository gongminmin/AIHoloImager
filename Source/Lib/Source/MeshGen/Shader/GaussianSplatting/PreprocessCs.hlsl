// Copyright (c) 2026 Minmin Gong
//

static const uint32_t BlockDim = 256;
static const float AlphaThreshold = 1 / 255.0f;

#define DATA_TYPE uint32_t
#include "../PrefixSumScanner/PrefixSumBlock.hlslh"

cbuffer param_cb
{
    uint32_t num_gaussians;
    uint32_t sh_degrees;
    uint32_t num_coeffs;
    float kernel_size;

    float4x4 view_mtx;
    float4x4 view_proj_mtx;

    float2 focal;
    float2 tan_fov;

    uint32_t2 width_height;
};

Buffer<float3> pos_buff;
Buffer<float3> scale_buff;
Buffer<float4> rotation_buff;
Buffer<float3> sh_buff;
Buffer<float> opacity_buff;

RWBuffer<float> color_buff;
RWBuffer<float4> conic_opacity_buff;
RWBuffer<float4> screen_pos_extents_buff;
RWBuffer<uint32_t> num_visible_gaussians_buff;
RWBuffer<uint32_t> visible_key_buff;
RWBuffer<uint32_t> visible_id_buff;

float2 Ndc2Screen(float4 ndc, uint32_t2 size)
{
    return (float2(ndc.x, -ndc.y) * 0.5f + 0.5f) * size;
}

template <typename T>
bool RectOverlap(T rect1, T rect2)
{
    return (rect1.x < rect2.z) && (rect1.z > rect2.x) && (rect1.y < rect2.w) && (rect1.w > rect2.y);
}

float4 ComputeCov2D(float3 pos, float2 focal, float2 tan_fov, float kernel_size, float cov_3d[6], float4x4 view_mtx)
{
    float4 t = mul(float4(pos, 1), view_mtx);
    t /= t.w;

    const float lim_x = 1.3f * tan_fov.x;
    const float lim_y = 1.3f * tan_fov.y;
    const float tx_tz = t.x / t.z;
    const float ty_tz = t.y / t.z;
    t.x = min(lim_x, max(-lim_x, tx_tz)) * t.z;
    t.y = min(lim_y, max(-lim_y, ty_tz)) * t.z;

    const float3x3 j_mtx = {
        focal.x / t.z, 0, -(focal.x * t.x) / (t.z * t.z),
        0, focal.y / t.z, -(focal.y * t.y) / (t.z * t.z),
        0, 0, 0,
    };

    const float3x3 w_mtx = transpose((float3x3)view_mtx);

    const float3x3 t_mtx = mul(j_mtx, w_mtx);

    const float3x3 vrk = {
        cov_3d[0], cov_3d[1], cov_3d[2],
        cov_3d[1], cov_3d[3], cov_3d[4],
        cov_3d[2], cov_3d[4], cov_3d[5],
    };

    // Eq 5
    const float3x3 cov = mul(mul(t_mtx, vrk), transpose(t_mtx));

    const float det_0 = max(1e-6f, cov[0].x * cov[1].y - cov[0].y * cov[0].y);
    const float det_1 = max(1e-6f, (cov[0].x + kernel_size) * (cov[1].y + kernel_size) - cov[0].y * cov[0].y);
    float coef = sqrt(det_0 / (det_1 + 1e-6f) + 1e-6f);
    if ((det_0 <= 1e-6f) || (det_1 <= 1e-6f))
    {
        coef = 0.0f;
    }

    return float4(cov[0].x + kernel_size, cov[0].y, cov[1].y + kernel_size, coef);
}

void ComputeCov3D(float3 scale, float4 rot, out float cov_3d[6])
{
    const float3x3 scaling_mtx = {
        scale.x, 0, 0,
        0, scale.y, 0,
        0, 0, scale.z,
    };

    rot = normalize(rot).yzwx;
    const float3x3 rot_mtx = {
        1 - 2 * (rot.y * rot.y + rot.z * rot.z), 2 * (rot.x * rot.y - rot.w * rot.z), 2 * (rot.x * rot.z + rot.w * rot.y),
        2 * (rot.x * rot.y + rot.w * rot.z), 1 - 2 * (rot.x * rot.x + rot.z * rot.z), 2 * (rot.y * rot.z - rot.w * rot.x),
        2 * (rot.x * rot.z - rot.w * rot.y), 2 * (rot.y * rot.z + rot.w * rot.x), 1 - 2 * (rot.x * rot.x + rot.y * rot.y),
    };

    // Eq 6
    const float3x3 mtx = mul(rot_mtx, scaling_mtx);
    const float3x3 sigma = mul(mtx, transpose(mtx));

    cov_3d[0] = sigma[0].x;
    cov_3d[1] = sigma[0].y;
    cov_3d[2] = sigma[0].z;
    cov_3d[3] = sigma[1].y;
    cov_3d[4] = sigma[1].z;
    cov_3d[5] = sigma[2].z;
}

float3 ComputeColorFromSh(uint32_t index, uint32_t degrees, uint32_t num_coeffs, float3 pos, Buffer<float3> sh_buff)
{
    static const float Sh_C0 = 0.28209479177387814f;

    const float3 sh = sh_buff[index * num_coeffs + 0];
    return max(Sh_C0 * sh + 0.5f, 0);
}

groupshared uint32_t group_offset;

[numthreads(BlockDim, 1, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID, uint32_t group_index : SV_GroupIndex)
{
    uint32_t visible = 0;
    float depth;
    float4 screen_pos_extents;
    float3 color;
    float4 conic_opacity;

    const uint32_t index = dtid.x;

    if (index == 0)
    {
        num_visible_gaussians_buff[1] = 1;
    }

    [branch]
    if (index < num_gaussians)
    {
        const float3 pos_os = pos_buff[index];
        const float4 pos_ps = mul(float4(pos_os, 1), view_proj_mtx);

        depth = pos_ps.w;

        [branch]
        if ((depth > 0.2f) && all(bool4(abs(pos_ps.xy) < abs(1.3f * pos_ps.w), pos_ps.z >= 0, pos_ps.z <= pos_ps.w)))
        {
            float cov_3d[6];
            ComputeCov3D(scale_buff[index], rotation_buff[index], cov_3d);

            const float4 cov = ComputeCov2D(pos_os, focal, tan_fov, kernel_size, cov_3d, view_mtx);

            const float det = cov.x * cov.z - cov.y * cov.y;
            [branch]
            if (det > 0)
            {
                const float mid = 0.5f * (cov.x + cov.z);
                const float extent = sqrt(max(0.1f, mid * mid - det));
                const float lambda_1 = mid + extent;
                const float lambda_2 = mid - extent;
                const float radius = 3 * sqrt(max(lambda_1, lambda_2));
                [branch]
                if (radius > 1e-6f)
                {
                    const float opacity = opacity_buff[index] * cov.w;
                    const float inv_coeff_low = opacity / AlphaThreshold;
                    [branch]
                    if (inv_coeff_low >= 1)
                    {
                        // "AdR-Gaussian: Accelerating Gaussian Splatting with Adaptive Radius" by Xinzhe Wang et al., 2024, Eq 10
                        const float coeff_ln = 2 * log(inv_coeff_low);
                        const float2 xy_max = sqrt(cov.xz * coeff_ln);
                        const float2 adaptive_radius = min(radius, xy_max);

                        const float2 screen_pos = Ndc2Screen(pos_ps / pos_ps.w, width_height);

                        [branch]
                        if (RectOverlap(float4(screen_pos - adaptive_radius, screen_pos + adaptive_radius), float4(0, 0, width_height)))
                        {
                            visible = 1;

                            screen_pos_extents = float4(screen_pos, adaptive_radius);
                            color = ComputeColorFromSh(index, sh_degrees, num_coeffs, pos_os, sh_buff);
                            conic_opacity = float4(cov.zyx / det, opacity);
                        }
                    }
                }
            }
        }
    }

    const uint32_t prefix_sum = ScanBlock(group_index, visible);
    if (group_index == BlockDim - 1)
    {
        InterlockedAdd(num_visible_gaussians_buff[0], prefix_sum + visible, group_offset);
    }
    GroupMemoryBarrierWithGroupSync();

    if (visible)
    {
        const uint32_t offset = group_offset + prefix_sum;
        visible_key_buff[offset] = asuint(depth);
        visible_id_buff[offset] = offset;

        screen_pos_extents_buff[offset] = screen_pos_extents;
        conic_opacity_buff[offset] = conic_opacity;
        color_buff[offset * 3 + 0] = color.x;
        color_buff[offset * 3 + 1] = color.y;
        color_buff[offset * 3 + 2] = color.z;
    }
}
