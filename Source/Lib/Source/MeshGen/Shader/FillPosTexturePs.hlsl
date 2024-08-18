float4 main(float3 pos_os : TEXCOORD0) : SV_Target0
{
    return float4(pos_os, 1);
}
