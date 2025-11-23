// Copyright (c) 2025 Minmin Gong
//

#include "Platform.hlslh"

cbuffer param_cb
{
    float4x4 mvp;
};

void main(float3 pos : POSITION,
          out float4 out_pos : SV_Position)
{
    out_pos = mul(float4(pos, 1), mvp);

    AdjustYDir(out_pos);
}
