// Copyright (c) 2026 Minmin Gong
//

#include "Atomic.hlslh"
#include "GSplatUtils.hlslh"

#ifdef __spirv__
[[vk::binding(6)]]
#endif
cbuffer param_cb
{
    float4 viewport;
    float3 bg_color;
    uint32_t2 rt_width_height;
};

#ifdef __spirv__
[[vk::binding(7)]]
#endif
Buffer<float3> grad_image_buff;

#ifdef __spirv__
[[vk::binding(8)]]
#endif
Texture2D gsplat_image;

#ifdef __spirv__
[[vk::binding(9)]]
#endif
RWBuffer<uint32_t> grad_color_buff;
#ifdef __spirv__
[[vk::binding(10)]]
#endif
RWBuffer<uint32_t> grad_screen_pos_buff;
#ifdef __spirv__
[[vk::binding(11)]]
#endif
RWBuffer<uint32_t> grad_conic_2d_buff;
#ifdef __spirv__
[[vk::binding(12)]]
#endif
RWBuffer<uint32_t> grad_opacities_buff;

#ifdef __spirv__
[[vk::binding(13)]]
#endif
RWTexture2D<float4> last_rgba_tex;
#ifdef __spirv__
[[vk::binding(14)]]
#endif
RasterizerOrderedTexture2D<float4> accum_color_alpha_tex;

// References
// https://ar5iv.labs.arxiv.org/html/2312.02121, Mathematical Supplement for the gsplat Library, chapter 3
// https://github.com/joeyan/gaussian_splatting/blob/main/MATH.md

void main(float3 screen_pos_point_id : TEXCOORD0,
          float3 color : TEXCOORD1,
          float4 conic_opacity : TEXCOORD2,
          float4 pos : SV_Position)
{
    if (IsHelperLane())
    {
        return;
    }

    const float2 screen_pos = screen_pos_point_id.xy;
    const uint32_t point_id = uint32_t(screen_pos_point_id.z);

    const uint32_t2 coord = uint32_t2(pos.xy);

    // Mapping from [0, W] to [-1, 1], resulting px ​= 0.5 *⋅(x + 1) * W. So dpx/dx = 0.5⋅* W. (and likewise for y)
    const float dpx_dx = +0.5f * viewport.z;
    const float dpy_dy = -0.5f * viewport.w;

    // Eq 4
    const float2 d = screen_pos - coord;
    const float power = -0.5f * (conic_opacity.x * d.x * d.x + conic_opacity.z * d.y * d.y) - conic_opacity.y * d.x * d.y;
    if (power <= 0)
    {
        // Eq 2
        const float g = exp(power);
        const float gaussian_alpha = min(0.99f, conic_opacity.w * g);
        [branch]
        if (gaussian_alpha >= AlphaThreshold)
        {
            float3 accum_color;
            float transparency;
            {
                const float4 accum_color_alpha = accum_color_alpha_tex[coord];
                const float4 last_rgba = last_rgba_tex[coord];

                accum_color = accum_color_alpha.rgb;
                transparency = accum_color_alpha.a;

                accum_color = lerp(accum_color, last_rgba.rgb, last_rgba.a);
                transparency /= 1 - gaussian_alpha;

                last_rgba_tex[coord] = float4(color, gaussian_alpha);
                accum_color_alpha_tex[coord] = float4(accum_color, transparency);
            }

            const float3 dl_dpixel = grad_image_buff[coord.y * rt_width_height.x + coord.x];

            const float dl_dalpha = dot(color - accum_color, dl_dpixel) * transparency
                + (-gsplat_image[coord].a / (1 - gaussian_alpha)) * dot(bg_color, dl_dpixel);

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
            AtomicAdd(grad_screen_pos_buff, point_id * 3 + 0, dl_dx);
            AtomicAdd(grad_screen_pos_buff, point_id * 3 + 1, dl_dy);

            // https://arxiv.org/pdf/2404.10772.pdf, Gaussian Opacity Fields: Efficient Adaptive Surface Reconstruction in Unbounded Scenes, Eq 15
            AtomicAdd(grad_screen_pos_buff, point_id * 3 + 2, abs(dl_dx) + abs(dl_dy));

            AtomicAdd(grad_conic_2d_buff, point_id * 4 + 0, -0.5f * dl_dg * gd.x * d.x);
            AtomicAdd(grad_conic_2d_buff, point_id * 4 + 1, +0.5f * dl_dg * gd.x * d.y);
            AtomicAdd(grad_conic_2d_buff, point_id * 4 + 3, -0.5f * dl_dg * gd.y * d.y);

            AtomicAdd(grad_opacities_buff, point_id, dl_dalpha * g);
        }
    }
}
