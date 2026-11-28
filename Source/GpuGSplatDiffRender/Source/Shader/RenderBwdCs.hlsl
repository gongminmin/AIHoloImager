// Copyright (c) 2026 Minmin Gong
//

#include "Atomic.hlslh"
#include "GSplatUtils.hlslh"

static const uint32_t BlockDim = 16;

cbuffer param_cb
{
    float4 viewport;
    float3 bg_color;
    uint32_t2 rt_width_height;
};

StructuredBuffer<float4> screen_pos_extents_buff;
StructuredBuffer<float4> conic_opacity_buff;
StructuredBuffer<float3> color_buff;
StructuredBuffer<float3> grad_image_buff;
Texture2D<float4> gsplat_image;
Texture2D<uint32_t> ppll_head;
StructuredBuffer<uint32_t2> ppll;

RWStructuredBuffer<uint32_t> grad_color_buff; // Float as uint due to atomic operations
RWStructuredBuffer<uint32_t> grad_screen_pos_buff; // Float as uint due to atomic operations
RWStructuredBuffer<uint32_t> grad_conic_2d_buff; // Float as uint due to atomic operations
RWStructuredBuffer<uint32_t> grad_opacities_buff; // Float as uint due to atomic operations

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    const uint32_t2 coord = dtid.xy;
    [branch]
    if (any(coord >= rt_width_height))
    {
        return;
    }

    const float transparency_final = gsplat_image[coord].a;
    float transparency = transparency_final;

    float3 accum_color = 0;
    const float3 dl_dpixel = grad_image_buff[coord.y * rt_width_height.x + coord.x];

    float4 last_rgba = 0;

    // Mapping from [0, W] to [-1, 1], resulting px ​= 0.5 *⋅(x + 1) * W. So dpx/dx = 0.5⋅* W. (and likewise for y)
    const float dpx_dx = +0.5f * viewport.z;
    const float dpy_dy = -0.5f * viewport.w;

    const float2 coord_vp = coord - viewport.xy;

    uint32_t ppll_next = ppll_head[coord];
    while (ppll_next != 0xFFFFFFFFU)
    {
        const uint32_t point_id = ppll[ppll_next].x;

        // Eq 4
        const float2 d = screen_pos_extents_buff[point_id].xy - coord_vp;
        const float4 conic_opacity = conic_opacity_buff[point_id];
        const float power = -0.5f * (conic_opacity.x * d.x * d.x + conic_opacity.z * d.y * d.y) - conic_opacity.y * d.x * d.y;
        [branch]
        if (power <= 0)
        {
            // Eq 2
            const float g = exp(power);
            const float gaussian_alpha = min(0.99f, conic_opacity.w * g);
            [branch]
            if (gaussian_alpha >= AlphaThreshold)
            {
                accum_color = lerp(accum_color, last_rgba.rgb, last_rgba.a);
                const float3 color = color_buff[point_id];
                last_rgba = float4(color, gaussian_alpha);

                transparency /= 1 - gaussian_alpha;

                const float dl_dalpha = dot(color - accum_color, dl_dpixel) * transparency
                    + (-transparency_final / (1 - gaussian_alpha)) * dot(bg_color, dl_dpixel);

                const float dpixel_dcolor = gaussian_alpha * transparency;
                const float3 dl_dcolor = dl_dpixel * dpixel_dcolor;
                [unroll]
                for (uint32_t ch = 0; ch < 3; ++ch)
                {
                    AtomicAdd(grad_color_buff, point_id * 3 + ch, dl_dcolor[ch]);
                }

                const float dl_dg = dl_dalpha * conic_opacity.w;
                const float2 gd = g * d;
                const float dg_dpx = -gd.x * conic_opacity.x - gd.y * conic_opacity.y;
                const float dg_dpy = -gd.y * conic_opacity.z - gd.x * conic_opacity.y;

                const float dl_dx = dl_dg * dg_dpx * dpx_dx;
                const float dl_dy = dl_dg * dg_dpy * dpy_dy;
                AtomicAdd(grad_screen_pos_buff, point_id * 2 + 0, dl_dx);
                AtomicAdd(grad_screen_pos_buff, point_id * 2 + 1, dl_dy);

                AtomicAdd(grad_conic_2d_buff, point_id * 3 + 0, -0.5f * dl_dg * gd.x * d.x);
                AtomicAdd(grad_conic_2d_buff, point_id * 3 + 1, +0.5f * dl_dg * gd.x * d.y);
                AtomicAdd(grad_conic_2d_buff, point_id * 3 + 2, -0.5f * dl_dg * gd.y * d.y);

                AtomicAdd(grad_opacities_buff, point_id, dl_dalpha * g);
            }
        }

        ppll_next = ppll[ppll_next].y;
    }
}
