// Copyright (c) 2024 Minmin Gong
//

#define BLOCK_DIM 16

cbuffer param_cb : register(b0)
{
    float3 k; // (0, 0), (0, 2), (1, 2)
    float3 params;
    float4 width_height;
};

Texture2D distorted_tex : register(t0);
SamplerState bilinear_sampler : register(s0);

RWTexture2D<unorm float4> undistorted_tex : register(u0);

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

[numthreads(BLOCK_DIM, BLOCK_DIM, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    float2 undistort_coord = dtid.xy;
    float2 distort_coord = GetDistortedCoord(undistort_coord);

    float4 color;
    if (all(bool4(distort_coord >= 0, distort_coord < width_height.xy)))
    {
        color = distorted_tex.SampleLevel(bilinear_sampler, distort_coord * width_height.zw, 0);
    }
    else
    {
        color = float4(0, 0, 0, 1);
    }

    undistorted_tex[dtid.xy] = color;
}
