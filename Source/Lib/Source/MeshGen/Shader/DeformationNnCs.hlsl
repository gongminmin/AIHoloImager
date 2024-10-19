// Copyright (c) 2024 Minmin Gong
//

#include "Nn.hlslh"
#include "MarchingCubesUtil.hlslh"

#define BLOCK_DIM 256
#define MAX_HIDDEN_DIM 64
#define MAX_FEATURES 240

cbuffer param_cb : register(b0)
{
    uint32_t num_samples;
    uint32_t num_features;

    uint32_t layer_1_nodes;
    uint32_t layer_2_nodes;
    uint32_t layer_3_nodes;
    uint32_t layer_4_nodes;

    uint32_t grid_res;
    uint32_t size;
    float grid_scale;
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

RWTexture3D<float4> density_deformations : register(u0);

[numthreads(BLOCK_DIM, 1, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (dtid.x >= num_samples)
    {
        return;
    }

    uint32_t3 coord = DecomposeCoord(dtid.x, size);
    float3 cube_vert = (float3(coord) / grid_res - 0.5f) * grid_scale * 0.5f + 0.5f;

    Tensor<MAX_FEATURES> inputs;
    uint32_t num_per_plane_features = num_features / 3;
    for (uint32_t i = 0; i < num_per_plane_features; ++i)
    {
        uint32_t index = i;
        inputs.Write(index, planes.SampleLevel(bilinear_sampler, float3(cube_vert.xy, index), 0));
        index += num_per_plane_features;
        inputs.Write(index, planes.SampleLevel(bilinear_sampler, float3(cube_vert.xz, index), 0));
        index += num_per_plane_features;
        inputs.Write(index, planes.SampleLevel(bilinear_sampler, float3(cube_vert.zy, index), 0));
    }

    Tensor<MAX_HIDDEN_DIM> nodes[2];
    {
        TensorView weights = TensorView::Create(nn_layer_1_weight, 0, uint32_t2(num_features, layer_1_nodes));
        TensorView biases = TensorView::Create(nn_layer_1_bias, 0, uint32_t2(layer_1_nodes, 1));

        Linear(nodes[0], inputs, weights, biases);
        ReLU(nodes[0], nodes[0]);
    }
    {
        TensorView weights = TensorView::Create(nn_layer_2_weight, 0, uint32_t2(layer_1_nodes, layer_2_nodes));
        TensorView biases = TensorView::Create(nn_layer_2_bias, 0, uint32_t2(layer_2_nodes, 1));

        Linear(nodes[1], nodes[0], weights, biases);
        ReLU(nodes[1], nodes[1]);
    }
    {
        TensorView weights = TensorView::Create(nn_layer_3_weight, 0, uint32_t2(layer_2_nodes, layer_3_nodes));
        TensorView biases = TensorView::Create(nn_layer_3_bias, 0, uint32_t2(layer_3_nodes, 1));

        Linear(nodes[0], nodes[1], weights, biases);
        ReLU(nodes[0], nodes[0]);
    }
    {
        TensorView weights = TensorView::Create(nn_layer_4_weight, 0, uint32_t2(layer_3_nodes, layer_4_nodes));
        TensorView biases = TensorView::Create(nn_layer_4_bias, 0, uint32_t2(layer_4_nodes, 1));

        Linear(nodes[1], nodes[0], weights, biases);
    }

    {
        float3 deformation = float3(nodes[1].Read(0), nodes[1].Read(1), nodes[1].Read(2));

        // Normalize the deformation to avoid the flipped triangles.
        const float DeformationMultiplier = 4.0f;
        deformation = 1.0 / (grid_res * DeformationMultiplier) * tanh(deformation);
        deformation *= size;

        density_deformations[coord.zyx].yzw = deformation;
    }
}
