// Copyright (c) 2025 Minmin Gong
//

void main(float4 pos : POSITION,
          out float4 out_pos : POSITION)
{
    out_pos = float4(pos.x, -pos.y, pos.zw);
}
