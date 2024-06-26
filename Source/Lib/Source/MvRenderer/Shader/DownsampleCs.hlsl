#define BLOCK_DIM 16
#define SSAA_SCALE 4

Texture2D ssaa_tex : register(t0);

RWTexture2D<unorm float4> tex : register(u0);

[numthreads(BLOCK_DIM, BLOCK_DIM, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    float4 color = 0;
    for (uint dy = 0; dy < SSAA_SCALE; ++dy)
    {
        for (uint dx = 0; dx < SSAA_SCALE; ++dx)
        {
            uint2 coord = dtid.xy * SSAA_SCALE + uint2(dx, dy);
            color += ssaa_tex.Load(uint3(coord, 0));
        }
    }

    color /= (SSAA_SCALE * SSAA_SCALE);
    tex[dtid.xy] = float4(color.rgb * color.a + (1 - color.a), 1);
}
