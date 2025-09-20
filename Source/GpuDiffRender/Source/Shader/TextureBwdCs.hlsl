// Copyright (c) 2025 Minmin Gong
//

#include "Common.hlslh"

static const uint32_t BlockDim = 16;

cbuffer param_cb : register(b0)
{
    uint32_t2 gbuffer_size;
    uint32_t2 tex_size;
    uint32_t num_channels;
    bool min_mag_filter_linear;
    bool mip_filter_linear;
    uint32_t mip_levels;
    uint32_t address_mode;
    uint32_t4 mip_level_offsets[4];
};

Texture2D texture : register(t0);
Buffer<float2> uv_buff : register(t1);
Texture2D grad_image : register(t2);
#if ENABLE_MIP
Buffer<float4> derivative_uv_buff : register(t3);
#endif

RWBuffer<uint32_t> grad_texture : register(u0);
RWBuffer<float2> grad_uv : register(u1);
#if ENABLE_MIP
RWBuffer<float4> grad_derivative_uv : register(u2);
#endif

float Wrap(float f, uint32_t size)
{
    return fmod(f, size);
}

float Mirror(float f, uint32_t size)
{
    f /= size;
    const float f_floor = floor(f);
    f = abs(f - 2 * f_floor);
    f = f > 1 ? (2 - f) : f;
    return f * size;
}

float Clamp(float f, uint32_t size)
{
    return clamp(f, 0, size - 1);
}

float2 CalcAddress(float2 uv, uint32_t2 tex_size)
{
    if (address_mode == 0)
    {
        uv.x = Wrap(uv.x, tex_size.x);
        uv.y = Wrap(uv.y, tex_size.y);
    }
    else if (address_mode == 1)
    {
        uv.x = Mirror(uv.x, tex_size.x);
        uv.y = Mirror(uv.y, tex_size.y);
    }
    else
    {
        uv.x = Clamp(uv.x, tex_size.x);
        uv.y = Clamp(uv.y, tex_size.y);
    }

    return uv;
}

#if ENABLE_MIP
void CalcMipInfo(out uint32_t2 mips, out float2 mip_weights, float level, uint32_t mip_levels, bool linear_filter)
{
    if (linear_filter)
    {
        mips.x = floor(level);
        mips.y = min(mips.x + 1, mip_levels - 1);
        if (mips.x == mips.y)
        {
            mip_weights.x = 1;
            mip_weights.y = 0;
        }
        else
        {
            mip_weights.y = level - mips.x;
            mip_weights.x = 1 - mip_weights.y;
        }
    }
    else
    {
        mips.x = round(level);
        mips.y = mips.x;
        mip_weights.x = 1;
        mip_weights.y = 0;
    }
}
#endif

void ScatterGradientPoint(uint32_t2 grad_coord, float2 uv, uint32_t level, float mip_weight)
{
    const uint32_t2 level_size = MipLevelSize(tex_size, level);
    float2 level_uv = uv * level_size;

    level_uv = CalcAddress(level_uv, level_size);

    const uint32_t2 coord = uint32_t2(floor(level_uv));
    const uint32_t texel_offset = (coord.y * level_size.x + coord.x) * num_channels;
    for (uint32_t i = 0; i < num_channels; ++i)
    {
        const float dl_dg = grad_image[grad_coord][i];
        if (dl_dg != 0)
        {
            const float mip_dl_dg = dl_dg * mip_weight;
            AtomicAdd(grad_texture, MipLevelOffset(mip_level_offsets, level) + texel_offset + i, mip_dl_dg);
        }
    }
}

void ScatterGradientLinear(uint32_t2 grad_coord, float2 uv, uint32_t level, float mip_weight, inout float2 dl_duv)
{
    const uint32_t2 level_size = MipLevelSize(tex_size, level);
    const float2 level_uv = uv * level_size;

    uint32_t2 quad_coords[4];
    quad_coords[0] = uint32_t2(floor(level_uv - 0.5f));
    quad_coords[1] = quad_coords[0] + uint2(1, 0);
    quad_coords[2] = quad_coords[0] + uint2(0, 1);
    quad_coords[2] = quad_coords[0] + uint2(1, 1);

    uint32_t quad_offsets[4];
    float4 quad_texels[4];
    for (uint32_t i = 0; i < 4; ++i)
    {
        quad_coords[i] = CalcAddress(quad_coords[i], level_size);
        quad_offsets[i] = (quad_coords[i].y * level_size.x + quad_coords[i].x) * num_channels;

        quad_texels[i] = texture.Load(uint3(quad_coords[i], level));
    }

    const float2 weight = level_uv - (quad_coords[0] + 0.5f);
    const float4 quad_weights = float4(
        (1 - weight.x) * (1 - weight.y),
        weight.x * (1 - weight.y),
        (1 - weight.x) * weight.y,
        weight.x * weight.y
    );

    for (uint32_t i = 0; i < num_channels; ++i)
    {
        const float dl_dg = grad_image[grad_coord][i];
        if (dl_dg != 0)
        {
            const float mip_dl_dg = dl_dg * mip_weight;

            const float4 grad_quad_weights = mip_dl_dg * quad_weights;
            for (uint32_t j = 0; j < 4; ++j)
            {
                AtomicAdd(grad_texture, MipLevelOffset(mip_level_offsets, level) + quad_offsets[j] + i, grad_quad_weights[j]);
            }

            // dL/d{u,v} = dL/dg * dg/d{u,v} = dL/dg * dg/d{wx,wy} * d{wx,wy}/d{u,v}
            // g = (1 - wx) * (1 - wy) * c0 + wx * (1 - wy) * c1 + (1 - wx) * wy * c2 + wx * wy * c3
            // dg/dwx = (wy - 1) * c0 + (1 - wy) * c1 - wy * c2 + wy * c3
            //        = wy * (c0 - c1 - c2 + c3) - c0 + c1
            // dg/dwy = (wx - 1) * c0 - wx * c1 + (1 - wx) * c2 + wx * c3
            //        = wx * (c0 - c1 - c2 + c3) - c0 + c2
            // d{wx,wy}/d{u,v} = level_size

            const float4 channel = float4(quad_texels[0][i], quad_texels[1][i], quad_texels[2][i], quad_texels[3][i]);
            const float2 da_dwxy = weight.yx * (channel.x - channel.y - channel.z + channel.w) - channel.x + channel.yz;
            const float2 dwxy_duv = level_size;
            dl_duv += mip_dl_dg * da_dwxy * dwxy_duv;
        }
    }
}

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (any(dtid.xy >= gbuffer_size))
    {
        return;
    }

    const uint32_t index = dtid.y * gbuffer_size.x + dtid.x;
    const uint32_t pixel_offset = index * num_channels;

    bool zero_grad = true;
    for (uint32_t i = 0; i < num_channels; ++i)
    {
        if (grad_image[dtid.xy][i] != 0)
        {
            zero_grad = false;
            break;
        }
    }

    [branch]
    if (zero_grad)
    {
        return;
    }

    const float2 uv = uv_buff[index];

#if ENABLE_MIP
    const float level = CalcMipLevel(derivative_uv_buff[index], tex_size, mip_levels);

    uint32_t2 mips;
    float2 mip_weights;
    CalcMipInfo(mips, mip_weights, level, mip_levels, mip_filter_linear);
#else
    uint32_t2 mips = uint32_t2(0, 0);
    float2 mip_weights = float2(1, 0);
#endif

    if (min_mag_filter_linear)
    {
        float2 dl_duv = 0;

        ScatterGradientLinear(dtid.xy, uv, mips.x, mip_weights.x, dl_duv);
#if ENABLE_MIP
        if (mip_filter_linear)
        {
            ScatterGradientLinear(dtid.xy, uv, mips.y, mip_weights.y, dl_duv);
        }
#endif

        grad_uv[index] = dl_duv;
    }
    else
    {
        ScatterGradientPoint(dtid.xy, uv, mips.x, mip_weights.x);
#if ENABLE_MIP
        if (mip_filter_linear)
        {
            ScatterGradientPoint(dtid.xy, uv, mips.y, mip_weights.y);
        }
#endif

        // The gradient of uv is always 0 in point mode
    }

#if ENABLE_MIP
    // TODO: Figure out how to calculate grad_derivative_uv
#endif
}
