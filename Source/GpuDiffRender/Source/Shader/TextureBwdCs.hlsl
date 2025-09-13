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
    uint32_t address_mode;
};

Texture2D texture : register(t0);
Buffer<float2> uv_buff : register(t1);
Buffer<float> grad_image : register(t2);

RWBuffer<uint32_t> grad_texture : register(u0);
RWBuffer<float2> grad_vtx_uv : register(u1);

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
        if (grad_image[pixel_offset + i] != 0)
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

    float2 uv = uv_buff[index] * tex_size;

    if (min_mag_filter_linear)
    {
        uint32_t2 quad_coords[4];
        quad_coords[0] = uint32_t2(floor(uv - 0.5f));
        quad_coords[1] = quad_coords[0] + uint2(1, 0);
        quad_coords[2] = quad_coords[0] + uint2(0, 1);
        quad_coords[2] = quad_coords[0] + uint2(1, 1);

        uint32_t quad_offsets[4];
        float4 quad_texels[4];
        for (uint32_t i = 0; i < 4; ++i)
        {
            quad_coords[i] = CalcAddress(quad_coords[i], tex_size);
            quad_offsets[i] = (quad_coords[i].y * tex_size.x + quad_coords[i].x) * num_channels;

            quad_texels[i] = texture.Load(uint3(quad_coords[i], 0));
        }

        const float2 weight = uv - (quad_coords[0] + 0.5f);
        const float4 quad_weights = float4(
            (1 - weight.x) * (1 - weight.y),
            weight.x * (1 - weight.y),
            (1 - weight.x) * weight.y,
            weight.x * weight.y
        );

        float2 grad_uv = 0;
        for (uint32_t i = 0; i < num_channels; ++i)
        {
            const float dl_da = grad_image[pixel_offset + i];
            if (dl_da != 0)
            {
                const float4 grad_quad_weights = dl_da * quad_weights;
                for (uint32_t j = 0; j < 4; ++j)
                {
                    AtomicAdd(grad_texture, quad_offsets[j] + i, quad_weights[j]);
                }

                // dL/d{u,v} = dL/dA * dA/d{u,v} = dL/dA * dA/d{wx,wy} * d{wx,wy}/d{u,v}
                // A = (1 - wx) * (1 - wy) * c0 + wx * (1 - wy) * c1 + (1 - wx) * wy * c2 + wx * wy * c3
                // dA/dwx = (wy - 1) * c0 + (1 - wy) * c1 - wy * c2 + wy * c3
                //        = wy * (c0 - c1 - c2 + c3) - c0 + c1
                // dA/dwy = (wx - 1) * c0 - wx * c1 + (1 - wx) * c2 + wx * c3
                //        = wx * (c0 - c1 - c2 + c3) - c0 + c2
                // d{wx,wy}/d{u,v} = tex_size

                const float4 channel = float4(quad_texels[0][i], quad_texels[1][i], quad_texels[2][i], quad_texels[3][i]);
                const float2 da_dwxy = weight.yx * (channel.x - channel.y - channel.z + channel.w) - channel.x + channel.yz;
                const float2 dwxy_duv = tex_size;
                grad_uv += dl_da * da_dwxy * dwxy_duv;
            }
        }
        grad_vtx_uv[index] = grad_uv;
    }
    else
    {
        uv = CalcAddress(uv, tex_size);

        const uint32_t2 coord = uint32_t2(floor(uv));
        const uint32_t texel_offset = (coord.y * tex_size.x + coord.x) * num_channels;
        for (uint32_t i = 0; i < num_channels; ++i)
        {
            const float dl_da = grad_image[pixel_offset + i];
            if (dl_da != 0)
            {
                AtomicAdd(grad_texture, texel_offset + i, dl_da);
            }
        }

        // The gradient of uv is always 0 in point mode
    }
}
