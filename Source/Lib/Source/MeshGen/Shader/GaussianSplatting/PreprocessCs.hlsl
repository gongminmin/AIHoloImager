// Copyright (c) 2026 Minmin Gong
//

#include "GSplatUtils.hlslh"
#include "Utils.hlslh"

static const uint32_t BlockDim = 256;

cbuffer param_cb
{
    uint32_t num_gaussians;
    uint32_t sh_degrees;
    uint32_t num_coeffs;
    float kernel_size;

    float4x4 view_mtx;
    float4x4 proj_mtx;

    float2 focal;
    float2 tan_fov;

    uint32_t2 width_height;
    uint32_t2 tile_grid;
};

Buffer<float3> pos_buff;
Buffer<float3> scale_buff;
Buffer<float4> rotation_buff;
Buffer<float3> sh_buff;
Buffer<float> opacity_buff;

RWBuffer<uint32_t> radius_buff;
RWBuffer<uint32_t> tiles_touched_buff;
RWBuffer<float> color_buff;
RWBuffer<float4> conic_opacity_buff;
RWBuffer<float> depth_buff;
RWBuffer<float2> screen_pos_buff;

float2 Ndc2Screen(float4 ndc, uint32_t2 size)
{
    return (ndc.xy * 0.5f + 0.5f) * size - 0.5f;
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
        focal.x / t.z, 0.0f, -(focal.x * t.x) / (t.z * t.z),
        0.0f, focal.y / t.z, -(focal.y * t.y) / (t.z * t.z),
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
    const float3x3 cov = mul(mul(t_mtx, transpose(vrk)), transpose(t_mtx));

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

    const float4 q = normalize(rot);
    float r = q.x;
    float x = q.y;
    float y = q.z;
    float z = q.w;

    const float3x3 rot_mtx = {
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2.f * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y),
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
    float3 result = Sh_C0 * sh;
    result += 0.5f;

    return max(result, float3(0.0f, 0.0f, 0.0f));
}

[numthreads(BlockDim, 1, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (dtid.x >= num_gaussians)
    {
        return;
    }

    const uint32_t index = dtid.x;

    radius_buff[index] = 0;
    tiles_touched_buff[index] = 0;

    const float3 pos_os = pos_buff[index];

    float4 pos_es = mul(float4(pos_os, 1), view_mtx);
    float4 pos_ps = mul(pos_es, proj_mtx);
    pos_ps /= pos_ps.w;

    const float depth = -pos_es.z / pos_es.w;

    [branch]
    if ((depth <= 0.2f) || any(abs(pos_ps.xy) > 1.3f))
    {
        return;
    }

    float cov_3d[6];
    ComputeCov3D(scale_buff[index], rotation_buff[index], cov_3d);

    const float4 cov = ComputeCov2D(pos_os, focal, tan_fov, kernel_size, cov_3d, view_mtx);

    const float det = cov.x * cov.z - cov.y * cov.y;
    [branch]
    if (det == 0.0f)
    {
        return;
    }

    const float inv_det = 1 / det;
    const float3 conic = float3(cov.z, -cov.y, cov.x) * inv_det;

    const float mid = 0.5f * (cov.x + cov.z);
    const float extent = sqrt(max(0.1f, mid * mid - det));
    const float lambda_1 = mid + extent;
    const float lambda_2 = mid - extent;
    const float radius = ceil(3 * sqrt(max(lambda_1, lambda_2)));
    const float2 screen_pos = Ndc2Screen(pos_ps, width_height);
    uint32_t2 rect_min, rect_max;
    GetRect(screen_pos, radius, rect_min, rect_max, tile_grid);
    const uint32_t tiles_touched = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y);
    [branch]
    if (tiles_touched == 0)
    {
        return;
    }

    {
        const float3 result = ComputeColorFromSh(index, sh_degrees, num_coeffs, pos_os, sh_buff);
        color_buff[index * 3 + 0] = result.x;
        color_buff[index * 3 + 1] = result.y;
        color_buff[index * 3 + 2] = result.z;
    }

    depth_buff[index] = depth;
    radius_buff[index] = radius;
    screen_pos_buff[index] = screen_pos;
    conic_opacity_buff[index] = float4(conic.xyz, opacity_buff[index] * cov.w);
    tiles_touched_buff[index] = tiles_touched;
}
