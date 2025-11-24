// Copyright (c) 2024-2025 Minmin Gong
//

static const uint32_t BlockDim = 16;

cbuffer param_cb
{
    float3 k; // (0, 0), (0, 2), (1, 2)
    float3 params;
    float4 width_height;
};

Texture2D distorted_tex;
SamplerState bilinear_sampler;

#ifdef __spirv__
[[vk::image_format("rgba8")]]
#endif
RWTexture2D<unorm float4> undistorted_tex;

// From Pinhole_Intrinsic_Radial_K3

float Focal()
{
    return k.x;
}

float2 PrincipalPoint()
{
    return k.yz;
}

float2 Image2Camera(float2 p)
{
    return (p - PrincipalPoint()) / Focal();
}

float2 Camera2Image(float2 p)
{
    return Focal() * p + PrincipalPoint();
}

float2 AddDistortion(float2 p)
{
    float r2 = dot(p, p);
    float r4 = r2 * r2;
    float r6 = r4 * r2;
    float r_coeff = 1 + dot(params, float3(r2, r4, r6));

    return p * r_coeff;
}

float2 GetDistortedCoord(float2 p)
{
    return Camera2Image(AddDistortion(Image2Camera(p)));
}

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    float2 undistort_coord = dtid.xy;
    float2 distort_coord = GetDistortedCoord(undistort_coord);

    float4 color = float4(0, 0, 0, 1);
    if (all(bool4(distort_coord >= 0, distort_coord < width_height.xy)))
    {
        color.rgb = distorted_tex.SampleLevel(bilinear_sampler, distort_coord * width_height.zw, 0).rgb;
    }

    undistorted_tex[dtid.xy] = color;
}
