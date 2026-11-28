// Copyright (c) 2026 Minmin Gong
//

#ifdef __spirv__
[[vk::binding(1)]]
#endif
globallycoherent RasterizerOrderedTexture2D<uint32_t> ppll_head;
#ifdef __spirv__
[[vk::binding(2)]]
#endif
RWStructuredBuffer<uint32_t> ppll;
#ifdef __spirv__
[[vk::binding(3)]]
#endif
RWStructuredBuffer<uint32_t> ppll_counter;

void main(uint32_t point_id : TEXCOORD0,
          float4 pos : SV_Position)
{
    const uint32_t2 coord = uint32_t2(pos.xy);

    const uint32_t curr_head = ppll_head[coord];

    uint32_t new_node;
    InterlockedAdd(ppll_counter[0], 1, new_node);
    ppll[new_node * 2 + 0] = point_id;
    ppll[new_node * 2 + 1] = curr_head;

    ppll_head[coord] = new_node;
}
