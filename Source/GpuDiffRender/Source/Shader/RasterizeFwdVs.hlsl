// Copyright (c) 2025 Minmin Gong
//

#include "Platform.hlslh"

void main(float4 pos : POSITION,
          out float4 out_pos : POSITION)
{
    out_pos = pos;

    AdjustYDir(out_pos);
}
