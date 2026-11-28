// Copyright (c) 2026 Minmin Gong
//

#include "GSplatUtils.hlslh"
#include "Sh.hlslh"

static const uint32_t BlockDim = 256;

cbuffer param_cb
{
    uint32_t num_gaussians;
    uint32_t sh_degrees;
    uint32_t num_coeffs;
    float kernel_size;

    float4x4 mv_mtx;
    float4x4 mvp_mtx;

    float2 focal;
    float2 tan_fov;

    float3 eye_pos_os;
};

StructuredBuffer<float3> pos_buff;
StructuredBuffer<float3> scale_buff;
StructuredBuffer<float4> rotation_buff;
StructuredBuffer<float3> sh_buff;
StructuredBuffer<float4> screen_pos_extents_buff;
StructuredBuffer<float> cov_3d_buff;
StructuredBuffer<uint32_t> clamped_buff;
StructuredBuffer<float4> conic_opacity_buff;
StructuredBuffer<float2> grad_screen_pos_buff;
StructuredBuffer<float3> grad_color_buff;
StructuredBuffer<float3> grad_conic_2d_buff;

RWStructuredBuffer<float> grad_positions_buff;
RWStructuredBuffer<float> grad_scales_buff;
RWStructuredBuffer<float4> grad_rotations_buff;
RWStructuredBuffer<float> grad_shs_buff;
RWStructuredBuffer<float> grad_opacities_buff;

float3 DNormVDV(float3 v, float3 dv)
{
    const float sum2 = dot(v, v);
    const float inv_sum3 = 1 / sqrt(sum2 * sum2 * sum2);

    float3 ret;
    ret.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * inv_sum3;
    ret.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * inv_sum3;
    ret.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * inv_sum3;
    return ret;
}

void ComputeCov2D(float3 pos_os, float2 focal, float2 tan_fov, float kernel_size, float cov_3d[6], float4x4 mv_mtx, float3 dl_dconic, float combined_opacity, inout float grad_opacity, out float grad_cov_3d[6], out float3 grad_position)
{
    float4 t = mul(float4(pos_os, 1), mv_mtx);
    t /= t.w;

    const float lim_x = 1.3f * tan_fov.x;
    const float lim_y = 1.3f * tan_fov.y;
    const float tx_tz = t.x / t.z;
    const float ty_tz = t.y / t.z;
    t.x = min(lim_x, max(-lim_x, tx_tz)) * t.z;
    t.y = min(lim_y, max(-lim_y, ty_tz)) * t.z;

    const float x_grad_mul = ((tx_tz < -lim_x) || (tx_tz > lim_x)) ? 0 : 1;
    const float y_grad_mul = ((ty_tz < -lim_y) || (ty_tz > lim_y)) ? 0 : 1;

    const float3x3 j_mtx = {
        focal.x / t.z, 0, -(focal.x * t.x) / (t.z * t.z),
        0, focal.y / t.z, -(focal.y * t.y) / (t.z * t.z),
        0, 0, 0,
    };

    const float3x3 w_mtx = transpose((float3x3)mv_mtx);

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

    const float opacity = combined_opacity / (coef + 1e-6f);
    const float dl_dcoef = grad_opacity * opacity;
    const float dl_dsqrt_coef = dl_dcoef * 0.5f / (coef + 1e-6f);
    const float dl_ddet0 = dl_dsqrt_coef / (det_1 + 1e-6f);
    const float dl_ddet1 = dl_dsqrt_coef * det_0 * (-1 / (det_1 * det_1 + 1e-6f));
    const float dcoef_da = dl_ddet0 * cov[1].y + dl_ddet1 * (cov[1].y + kernel_size);
    const float dcoef_db = dl_ddet0 * (-2 * cov[0].y) + dl_ddet1 * (-2 * cov[0].y);
    const float dcoef_dc = dl_ddet0 * cov[0].x + dl_ddet1 * (cov[0].x + kernel_size);

    cov[0].x += kernel_size;
    cov[1].y += kernel_size;
    const float a = cov[0].x;
    const float b = cov[0].y;
    const float c = cov[1].y;

    float denom = a * c - b * b;
    float dl_da = 0;
    float dl_db = 0;
    float dl_dc = 0;
    if (denom != 0)
    {
        const float inv_denom2 = 1.0f / ((denom * denom) + 1e-6f);

        // dl/da = dl/d_conic_a * dconic_a/d_a
        dl_da = inv_denom2 * (-c * c * dl_dconic.x + 2 * b * c * dl_dconic.y + (denom - a * c) * dl_dconic.z);
        dl_dc = inv_denom2 * (-a * a * dl_dconic.z + 2 * a * b * dl_dconic.y + (denom - a * c) * dl_dconic.x);
        dl_db = inv_denom2 * 2 * (b * c * dl_dconic.x - (denom + 2 * b * b) * dl_dconic.y + a * b * dl_dconic.z);

        if ((det_0 <= 1e-6f) || (det_1 <= 1e-6f))
        {
            grad_opacity = 0;
        }
        else
        {
            dl_da += dcoef_da;
            dl_dc += dcoef_dc;
            dl_db += dcoef_db;

            grad_opacity *= coef;
        }

        grad_cov_3d[0] = t_mtx[0].x * t_mtx[0].x * dl_da + t_mtx[0].x * t_mtx[1].x * dl_db + t_mtx[1].x * t_mtx[1].x * dl_dc;
        grad_cov_3d[3] = t_mtx[0].y * t_mtx[0].y * dl_da + t_mtx[0].y * t_mtx[1].y * dl_db + t_mtx[1].y * t_mtx[1].y * dl_dc;
        grad_cov_3d[5] = t_mtx[0].z * t_mtx[0].z * dl_da + t_mtx[0].z * t_mtx[1].z * dl_db + t_mtx[1].z * t_mtx[1].z * dl_dc;

        grad_cov_3d[1] = 2 * t_mtx[0].x * t_mtx[0].y * dl_da + (t_mtx[0].x * t_mtx[1].y + t_mtx[0].y * t_mtx[1].x) * dl_db + 2 * t_mtx[1].x * t_mtx[1].y * dl_dc;
        grad_cov_3d[2] = 2 * t_mtx[0].x * t_mtx[0].z * dl_da + (t_mtx[0].x * t_mtx[1].z + t_mtx[0].z * t_mtx[1].x) * dl_db + 2 * t_mtx[1].x * t_mtx[1].z * dl_dc;
        grad_cov_3d[4] = 2 * t_mtx[0].z * t_mtx[0].y * dl_da + (t_mtx[0].y * t_mtx[1].z + t_mtx[0].z * t_mtx[1].y) * dl_db + 2 * t_mtx[1].y * t_mtx[1].z * dl_dc;
    }
    else
    {
        [unroll]
        for (uint32_t i = 0; i < 6; ++i)
        {
            grad_cov_3d[i] = 0;
        }
    }

    const float dl_dt00 = 2 * (t_mtx[0].x * vrk[0].x + t_mtx[0].y * vrk[0].y + t_mtx[0].z * vrk[0].z) * dl_da + (t_mtx[1].x * vrk[0].x + t_mtx[1].y * vrk[0].y + t_mtx[1].z * vrk[0].z) * dl_db;
    const float dl_dt01 = 2 * (t_mtx[0].x * vrk[1].x + t_mtx[0].y * vrk[1].y + t_mtx[0].z * vrk[1].z) * dl_da + (t_mtx[1].x * vrk[1].x + t_mtx[1].y * vrk[1].y + t_mtx[1].z * vrk[1].z) * dl_db;
    const float dl_dt02 = 2 * (t_mtx[0].x * vrk[2].x + t_mtx[0].y * vrk[2].y + t_mtx[0].z * vrk[2].z) * dl_da + (t_mtx[1].x * vrk[2].x + t_mtx[1].y * vrk[2].y + t_mtx[1].z * vrk[2].z) * dl_db;
    const float dl_dt10 = 2 * (t_mtx[1].x * vrk[0].x + t_mtx[1].y * vrk[0].y + t_mtx[1].z * vrk[0].z) * dl_dc + (t_mtx[0].x * vrk[0].x + t_mtx[0].y * vrk[0].y + t_mtx[0].z * vrk[0].z) * dl_db;
    const float dl_dt11 = 2 * (t_mtx[1].x * vrk[1].x + t_mtx[1].y * vrk[1].y + t_mtx[1].z * vrk[1].z) * dl_dc + (t_mtx[0].x * vrk[1].x + t_mtx[0].y * vrk[1].y + t_mtx[0].z * vrk[1].z) * dl_db;
    const float dl_dt12 = 2 * (t_mtx[1].x * vrk[2].x + t_mtx[1].y * vrk[2].y + t_mtx[1].z * vrk[2].z) * dl_dc + (t_mtx[0].x * vrk[2].x + t_mtx[0].y * vrk[2].y + t_mtx[0].z * vrk[2].z) * dl_db;

    const float dl_dj00 = w_mtx[0][0] * dl_dt00 + w_mtx[0][1] * dl_dt01 + w_mtx[0][2] * dl_dt02;
    const float dl_dj02 = w_mtx[2][0] * dl_dt00 + w_mtx[2][1] * dl_dt01 + w_mtx[2][2] * dl_dt02;
    const float dl_dj11 = w_mtx[1][0] * dl_dt10 + w_mtx[1][1] * dl_dt11 + w_mtx[1][2] * dl_dt12;
    const float dl_dj12 = w_mtx[2][0] * dl_dt10 + w_mtx[2][1] * dl_dt11 + w_mtx[2][2] * dl_dt12;

    const float tz = 1 / t.z;
    const float tz2 = tz * tz;
    const float tz3 = tz2 * tz;

    const float dl_dtx = x_grad_mul * -focal.x * tz2 * dl_dj02;
    const float dl_dty = y_grad_mul * -focal.y * tz2 * dl_dj12;
    const float dl_dtz = -focal.x * tz2 * dl_dj00 - focal.y * tz2 * dl_dj11 + (2 * focal.x * t.x) * tz3 * dl_dj02 + (2 * focal.y * t.y) * tz3 * dl_dj12;
    grad_position = mul(float3(dl_dtx, dl_dty, dl_dtz), w_mtx);
}

void ComputeColorFromSh(uint32_t index, uint32_t sh_degrees, float3 pos_os, float3 eye_pos_os, uint32_t3 clamped, inout float3 grad_position, RWStructuredBuffer<float> grad_shs_buff)
{
    const float3 dir_orig = pos_os - eye_pos_os;
    const float3 dir = normalize(dir_orig);

    const uint32_t sh_offset = index * num_coeffs;

    float basis[16];
    ShEvalBasis(sh_degrees, dir, basis);

    // Use PyTorch rule for clamping: if clamping was applied, gradient becomes 0
    const float3 dl_drgb = select(clamped, 0, grad_color_buff[index]);

    for (uint32_t c = 0; c < num_coeffs; ++c)
    {
        WriteFloat3(grad_shs_buff, sh_offset + c, basis[c] * dl_drgb);
    }

    float3 drgb_dx = 0;
    float3 drgb_dy = 0;
    float3 drgb_dz = 0;

    if (sh_degrees > 0)
    {
        drgb_dx = Sh_C1[2] * sh_buff[sh_offset + 3];
        drgb_dy = Sh_C1[0] * sh_buff[sh_offset + 1];
        drgb_dz = Sh_C1[1] * sh_buff[sh_offset + 2];

        if (sh_degrees > 1)
        {
            const float xx = dir.x * dir.x;
            const float yy = dir.y * dir.y;
            const float zz = dir.z * dir.z;
            const float xy = dir.x * dir.y;
            const float yz = dir.y * dir.z;
            const float xz = dir.x * dir.z;

            drgb_dx += Sh_C2[0] * dir.y * sh_buff[sh_offset + 4] + Sh_C2[2] * 2 * -dir.x * sh_buff[sh_offset + 6] + Sh_C2[3] * dir.z * sh_buff[sh_offset + 7] + Sh_C2[4] * 2 * dir.x * sh_buff[sh_offset + 8];
            drgb_dy += Sh_C2[0] * dir.x * sh_buff[sh_offset + 4] + Sh_C2[1] * dir.z * sh_buff[sh_offset + 5] + Sh_C2[2] * 2 * -dir.y * sh_buff[sh_offset + 6] + Sh_C2[4] * 2 * -dir.y * sh_buff[sh_offset + 8];
            drgb_dz += Sh_C2[1] * dir.y * sh_buff[sh_offset + 5] + Sh_C2[2] * 2 * 2 * dir.z * sh_buff[sh_offset + 6] + Sh_C2[3] * dir.x * sh_buff[sh_offset + 7];

            if (sh_degrees > 2)
            {
                drgb_dx += Sh_C3[0] * sh_buff[sh_offset + 9] * 3 * 2 * xy
                         + Sh_C3[1] * sh_buff[sh_offset + 10] * yz
                         + Sh_C3[2] * sh_buff[sh_offset + 11] * -2 * xy
                         + Sh_C3[3] * sh_buff[sh_offset + 12] * -3 * 2 * xz
                         + Sh_C3[4] * sh_buff[sh_offset + 13] * (-3 * xx + 4 * zz - yy)
                         + Sh_C3[5] * sh_buff[sh_offset + 14] * 2 * xz
                         + Sh_C3[6] * sh_buff[sh_offset + 15] * 3 * (xx - yy);
                drgb_dy += Sh_C3[0] * sh_buff[sh_offset + 9] * 3 * (xx - yy)
                         + Sh_C3[1] * sh_buff[sh_offset + 10] * xz
                         + Sh_C3[2] * sh_buff[sh_offset + 11] * (-3 * yy + 4 * zz - xx)
                         + Sh_C3[3] * sh_buff[sh_offset + 12] * -3 * 2 * yz
                         + Sh_C3[4] * sh_buff[sh_offset + 13] * -2 * xy
                         + Sh_C3[5] * sh_buff[sh_offset + 14] * -2 * yz
                         + Sh_C3[6] * sh_buff[sh_offset + 15] * -3 * 2 * xy;
                drgb_dz += Sh_C3[1] * sh_buff[sh_offset + 10] * xy
                         + Sh_C3[2] * sh_buff[sh_offset + 11] * 4 * 2 * yz
                         + Sh_C3[3] * sh_buff[sh_offset + 12] * 3 * (2 * zz - xx - yy)
                         + Sh_C3[4] * sh_buff[sh_offset + 13] * 4 * 2 * xz
                         + Sh_C3[5] * sh_buff[sh_offset + 14] * (xx - yy);
            }
        }
    }

    const float3 dl_ddir = float3(dot(dl_drgb, drgb_dx), dot(dl_drgb, drgb_dy), dot(dl_drgb, drgb_dz));
    grad_position += DNormVDV(dir_orig, dl_ddir);
}

void ComputeCov3D(float3 scale, float4 rot, float dl_dcov_3d[6], out float3 grad_scale, out float4 grad_rotation)
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

    const float3x3 dl_dsigma = {
        dl_dcov_3d[0], 0.5f * dl_dcov_3d[1], 0.5f * dl_dcov_3d[2],
        0.5f * dl_dcov_3d[1], dl_dcov_3d[3], 0.5f * dl_dcov_3d[4],
        0.5f * dl_dcov_3d[2], 0.5f * dl_dcov_3d[4], dl_dcov_3d[5]
    };

    const float3x3 rot_mtx_t = transpose(rot_mtx);
    float3x3 dl_dm_t = transpose(2 * mul(dl_dsigma, mtx));
    grad_scale = float3(dot(rot_mtx_t[0], dl_dm_t[0]), dot(rot_mtx_t[1], dl_dm_t[1]), dot(rot_mtx_t[2], dl_dm_t[2]));

    dl_dm_t[0] *= scale.x;
    dl_dm_t[1] *= scale.y;
    dl_dm_t[2] *= scale.z;

    grad_rotation.x = 2 * rot.z * (dl_dm_t[0].y - dl_dm_t[1].x) + 2 * rot.y * (dl_dm_t[2].x - dl_dm_t[0].z) + 2 * rot.x * (dl_dm_t[1].z - dl_dm_t[2].y);
    grad_rotation.y = 2 * rot.y * (dl_dm_t[1].x + dl_dm_t[0].y) + 2 * rot.z * (dl_dm_t[2].x + dl_dm_t[0].z) + 2 * rot.w * (dl_dm_t[1].z - dl_dm_t[2].y) - 4 * rot.x * (dl_dm_t[2].z + dl_dm_t[1].y);
    grad_rotation.z = 2 * rot.x * (dl_dm_t[1].x + dl_dm_t[0].y) + 2 * rot.w * (dl_dm_t[2].x - dl_dm_t[0].z) + 2 * rot.z * (dl_dm_t[1].z + dl_dm_t[2].y) - 4 * rot.y * (dl_dm_t[2].z + dl_dm_t[0].x);
    grad_rotation.w = 2 * rot.w * (dl_dm_t[0].y - dl_dm_t[1].x) + 2 * rot.x * (dl_dm_t[2].x + dl_dm_t[0].z) + 2 * rot.y * (dl_dm_t[1].z + dl_dm_t[2].y) - 4 * rot.z * (dl_dm_t[1].y + dl_dm_t[0].x);
}

[numthreads(BlockDim, 1, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    const uint32_t index = dtid.x;

    [branch]
    if ((index >= num_gaussians) || any(screen_pos_extents_buff[index] <= 0))
    {
        return;
    }

    const float3 pos_os = pos_buff[index];

    float cov_3d[6];
    [unroll]
    for (uint32_t i = 0; i < 6; ++i)
    {
        cov_3d[i] = cov_3d_buff[index * 6 + i];
    }

    float grad_opacity = grad_opacities_buff[index];
    float grad_cov_3d[6];
    float3 grad_position;
    ComputeCov2D(pos_os, focal, tan_fov, kernel_size, cov_3d, mv_mtx, grad_conic_2d_buff[index], conic_opacity_buff[index].w, grad_opacity, grad_cov_3d, grad_position);

    grad_opacities_buff[index] = grad_opacity;

    const float4 pos_ps = mul(float4(pos_os, 1), mvp_mtx);
    const float inv_w = 1 / (pos_ps.w + 1e-6f);

    const float mul1 = (dot(pos_os, float3(mvp_mtx[0].x, mvp_mtx[1].x, mvp_mtx[2].x)) + mvp_mtx[3].x) * inv_w * inv_w;
    const float mul2 = (dot(pos_os, float3(mvp_mtx[0].y, mvp_mtx[1].y, mvp_mtx[2].y)) + mvp_mtx[3].y) * inv_w * inv_w;
    const float2 grad_screen_pos = grad_screen_pos_buff[index];
    const float3 dl_dpos = {
        grad_screen_pos.x * (mvp_mtx[0].x * inv_w - mvp_mtx[0].w * mul1) + grad_screen_pos.y * (mvp_mtx[0].y * inv_w - mvp_mtx[0].w * mul2),
        grad_screen_pos.x * (mvp_mtx[1].x * inv_w - mvp_mtx[1].w * mul1) + grad_screen_pos.y * (mvp_mtx[1].y * inv_w - mvp_mtx[1].w * mul2),
        grad_screen_pos.x * (mvp_mtx[2].x * inv_w - mvp_mtx[2].w * mul1) + grad_screen_pos.y * (mvp_mtx[2].y * inv_w - mvp_mtx[2].w * mul2),
    };
    grad_position += dl_dpos;

    const uint32_t3 clamped = (clamped_buff[index] >> uint32_t3(0, 1, 2)) & 1U;
    ComputeColorFromSh(index, sh_degrees, pos_os, eye_pos_os, clamped, grad_position, grad_shs_buff);
    WriteFloat3(grad_positions_buff, index, grad_position);

    float3 grad_scale;
    float4 grad_rotation;
    ComputeCov3D(scale_buff[index], rotation_buff[index], grad_cov_3d, grad_scale, grad_rotation);
    WriteFloat3(grad_scales_buff, index, grad_scale);
    grad_rotations_buff[index] = grad_rotation;
}
