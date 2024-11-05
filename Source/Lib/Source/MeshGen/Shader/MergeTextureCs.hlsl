// Copyright (c) 2024 Minmin Gong
//

#include "Nn.hlslh"
#include "TriplaneNeRF.hlslh"

static const uint32_t BlockDim = 16;
static const uint32_t MaxHiddenDim = 64;
static const uint32_t MaxFeatures = 240;

cbuffer param_cb : register(b0)
{
    uint32_t num_samples;
    uint32_t num_features;

    uint32_t layer_1_nodes;
    uint32_t layer_2_nodes;
    uint32_t layer_3_nodes;
    uint32_t layer_4_nodes;

    uint32_t texture_size;
    float4x4 inv_model;
};

SamplerState bilinear_sampler : register(s0);

Texture2DArray<float> planes : register(t0);

Buffer<float> nn_layer_1_weight : register(t1);
Buffer<float> nn_layer_1_bias : register(t2);
Buffer<float> nn_layer_2_weight : register(t3);
Buffer<float> nn_layer_2_bias : register(t4);
Buffer<float> nn_layer_3_weight : register(t5);
Buffer<float> nn_layer_3_bias : register(t6);
Buffer<float> nn_layer_4_weight : register(t7);
Buffer<float> nn_layer_4_bias : register(t8);

Texture2D pos_tex : register(t9);

RWTexture2D<unorm float4> merged_tex : register(u0);

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (any(dtid.xy >= texture_size))
    {
        return;
    }

    [branch]
    if (merged_tex[dtid.xy].a > 0.5f)
    {
        return;
    }

    float4 pos_ws = pos_tex[dtid.xy];
    [branch]
    if (pos_ws.a < 0.5f)
    {
        return;
    }

    Tensor<MaxHiddenDim> nodes[2];
    {
        float4 pos_os = mul(pos_ws, inv_model);
        pos_os /= pos_os.w;

        Tensor<MaxFeatures> inputs;
        FetchTriplaneNeRF(inputs, planes, bilinear_sampler, pos_os.xyz, num_features);

        TensorView weight = TensorView::Create(nn_layer_1_weight, 0, uint32_t2(num_features, layer_1_nodes));
        TensorView bias = TensorView::Create(nn_layer_1_bias, 0, uint32_t2(layer_1_nodes, 1));

        Linear(nodes[0], inputs, weight, bias);
        ReLU(nodes[0], nodes[0]);
    }
    {
        TensorView weight = TensorView::Create(nn_layer_2_weight, 0, uint32_t2(layer_1_nodes, layer_2_nodes));
        TensorView bias = TensorView::Create(nn_layer_2_bias, 0, uint32_t2(layer_2_nodes, 1));

        Linear(nodes[1], nodes[0], weight, bias);
        ReLU(nodes[1], nodes[1]);
    }
    {
        TensorView weight = TensorView::Create(nn_layer_3_weight, 0, uint32_t2(layer_2_nodes, layer_3_nodes));
        TensorView bias = TensorView::Create(nn_layer_3_bias, 0, uint32_t2(layer_3_nodes, 1));

        Linear(nodes[0], nodes[1], weight, bias);
        ReLU(nodes[0], nodes[0]);
    }
    {
        TensorView weight = TensorView::Create(nn_layer_4_weight, 0, uint32_t2(layer_3_nodes, layer_4_nodes));
        TensorView bias = TensorView::Create(nn_layer_4_bias, 0, uint32_t2(layer_4_nodes, 1));

        Linear(nodes[1], nodes[0], weight, bias);
        Sigmoid(nodes[1], nodes[1]);
    }

    // layer_4_nodes == 3
    {
        merged_tex[dtid.xy] = float4(saturate(float3(nodes[1].Read(0), nodes[1].Read(1), nodes[1].Read(2))), 1);
    }
}
