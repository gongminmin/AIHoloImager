// Copyright (c) 2026 Minmin Gong
//

#include "Sorter.hpp"

#include "Base/ErrorHandling.hpp"
#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuResourceViews.hpp"

#include "CompiledShader/MeshGen/Sorter/DownsweepCs_Uint2_Uint.h"
#include "CompiledShader/MeshGen/Sorter/DownsweepCs_Uint_Uint.h"
#include "CompiledShader/MeshGen/Sorter/ScanCs_Uint2_Uint.h"
#include "CompiledShader/MeshGen/Sorter/ScanCs_Uint_Uint.h"
#include "CompiledShader/MeshGen/Sorter/UpsweepCs_Uint2_Uint.h"
#include "CompiledShader/MeshGen/Sorter/UpsweepCs_Uint_Uint.h"

namespace AIHoloImager
{
    class Sorter::Impl
    {
    public:
        explicit Impl(AIHoloImagerInternal& aihi) : gpu_system_(aihi.GpuSystemInstance())
        {
            {
                const ShaderInfo shader = {DEFINE_SHADER(UpsweepCs_Uint2_Uint)};
                upsweep_uint2_uint_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }
            {
                const ShaderInfo shader = {DEFINE_SHADER(UpsweepCs_Uint_Uint)};
                upsweep_uint_uint_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }
            {
                const ShaderInfo shader = {DEFINE_SHADER(ScanCs_Uint2_Uint)};
                scan_uint2_uint_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }
            {
                const ShaderInfo shader = {DEFINE_SHADER(ScanCs_Uint_Uint)};
                scan_uint_uint_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }
            {
                const ShaderInfo shader = {DEFINE_SHADER(DownsweepCs_Uint2_Uint)};
                downsweep_uint2_uint_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }
            {
                const ShaderInfo shader = {DEFINE_SHADER(DownsweepCs_Uint_Uint)};
                downsweep_uint_uint_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }
        }

        void RadixSort(GpuCommandList& cmd_list, const GpuBuffer& keys, GpuFormat key_format, const GpuBuffer& values,
            GpuFormat value_format, uint32_t num_items, GpuBuffer& sorted_keys, GpuBuffer& sorted_values, uint32_t bits)
        {
            constexpr uint32_t NumDigitBins = 256;
            constexpr uint32_t WorksetSize = 512;
            constexpr uint32_t PartitionDivision = 8;
            constexpr uint32_t PartitionSize = PartitionDivision * WorksetSize;

            const uint32_t num_partitions = DivUp(num_items, PartitionSize);
            const uint32_t key_size = FormatSize(key_format);
            const uint32_t value_size = FormatSize(value_format);

            bits = std::min(bits, key_size * 8);

            GpuComputePipeline* upsweep_pipeline;
            GpuComputePipeline* scan_pipeline;
            GpuComputePipeline* downsweep_pipeline;
            switch (key_format)
            {
            case GpuFormat::R32_Uint:
                upsweep_pipeline = &upsweep_uint_uint_pipeline_;
                scan_pipeline = &scan_uint_uint_pipeline_;
                downsweep_pipeline = &downsweep_uint_uint_pipeline_;
                break;
            case GpuFormat::RG32_Uint:
                upsweep_pipeline = &upsweep_uint2_uint_pipeline_;
                scan_pipeline = &scan_uint2_uint_pipeline_;
                downsweep_pipeline = &downsweep_uint2_uint_pipeline_;
                break;

            default:
                Unreachable("Unsupported key format");
            }

            GpuBuffer ping_pong_keys[2];
            GpuBuffer ping_pong_values[2];
            for (size_t i = 0; i < std::size(ping_pong_keys); ++i)
            {
                ping_pong_keys[i] = GpuBuffer(gpu_system_, num_items * key_size, GpuHeap::Default, GpuResourceFlag::UnorderedAccess,
                    std::format("Sorter.ping_pong_keys_{}", i));
                ping_pong_values[i] = GpuBuffer(gpu_system_, num_items * value_size, GpuHeap::Default, GpuResourceFlag::UnorderedAccess,
                    std::format("Sorter.ping_pong_values_{}", i));
            }

            GpuBuffer global_histogram(gpu_system_, key_size * NumDigitBins * sizeof(uint32_t), GpuHeap::Default,
                GpuResourceFlag::UnorderedAccess, "Sorter.global_histogram");
            const GpuShaderResourceView global_histogram_srv(gpu_system_, global_histogram, GpuFormat::R32_Uint);
            GpuUnorderedAccessView global_histogram_uav(gpu_system_, global_histogram, GpuFormat::R32_Uint);

            {
                const uint32_t clear_clr[] = {0, 0, 0, 0};
                cmd_list.Clear(global_histogram_uav, clear_clr);
            }

            GpuBuffer partition_histogram(gpu_system_, num_partitions * NumDigitBins * sizeof(uint32_t), GpuHeap::Default,
                GpuResourceFlag::UnorderedAccess, "Sorter.partition_histogram");
            const GpuShaderResourceView partition_histogram_srv(gpu_system_, partition_histogram, GpuFormat::R32_Uint);
            GpuUnorderedAccessView partition_histogram_uav(gpu_system_, partition_histogram, GpuFormat::R32_Uint);

            const uint32_t num_passes = DivUp(bits, 8);
            for (uint32_t pass = 0; pass < num_passes; ++pass)
            {
                const uint32_t input_index = pass & 1;
                const uint32_t output_index = (input_index + 1) & 1;

                const uint32_t pass_begin_bit = pass * 8;
                const uint32_t pass_end_bit = std::min(bits, pass_begin_bit + 8);

                GpuShaderResourceView keys_srv;
                GpuShaderResourceView values_srv;
                if (pass == 0)
                {
                    keys_srv = GpuShaderResourceView(gpu_system_, keys, key_format);
                    values_srv = GpuShaderResourceView(gpu_system_, values, value_format);
                }
                else
                {
                    keys_srv = GpuShaderResourceView(gpu_system_, ping_pong_keys[input_index], key_format);
                    values_srv = GpuShaderResourceView(gpu_system_, ping_pong_values[input_index], value_format);
                }

                {
                    constexpr uint32_t BlockDim = 256;

                    GpuConstantBufferOfType<UpsweepConstantBuffer> upsweep_cb(gpu_system_, "upsweep_cb");
                    upsweep_cb->num_items = num_items;
                    upsweep_cb->pass = pass;
                    upsweep_cb->base_shift = pass_begin_bit;
                    upsweep_cb->bits = pass_end_bit - pass_begin_bit;
                    upsweep_cb.UploadStaging();

                    const GpuConstantBufferView upsweep_cbv(gpu_system_, upsweep_cb);

                    std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                        {"param_cb", &upsweep_cbv},
                    };
                    std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                        {"keys", &keys_srv},
                    };
                    std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                        {"global_hist", &global_histogram_uav},
                        {"partition_hist", &partition_histogram_uav},
                    };
                    const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
                    cmd_list.Compute(*upsweep_pipeline, DivUp(num_partitions * WorksetSize, BlockDim), 1, 1, shader_binding);
                }
                {
                    GpuConstantBufferOfType<ScanConstantBuffer> scan_cb(gpu_system_, "scan_cb");
                    scan_cb->num_items = num_items;
                    scan_cb->pass = pass;
                    scan_cb.UploadStaging();

                    const GpuConstantBufferView scan_cbv(gpu_system_, scan_cb);

                    std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                        {"param_cb", &scan_cbv},
                    };
                    std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                        {"global_hist", &global_histogram_uav},
                        {"partition_hist", &partition_histogram_uav},
                    };
                    const GpuCommandList::ShaderBinding shader_binding = {cbvs, {}, uavs};
                    cmd_list.Compute(*scan_pipeline, NumDigitBins, 1, 1, shader_binding);
                }
                {
                    GpuUnorderedAccessView output_keys_uav;
                    GpuUnorderedAccessView output_values_uav;
                    if (pass == num_passes - 1)
                    {
                        output_keys_uav = GpuUnorderedAccessView(gpu_system_, sorted_keys, key_format);
                        output_values_uav = GpuUnorderedAccessView(gpu_system_, sorted_values, value_format);
                    }
                    else
                    {
                        output_keys_uav = GpuUnorderedAccessView(gpu_system_, ping_pong_keys[output_index], key_format);
                        output_values_uav = GpuUnorderedAccessView(gpu_system_, ping_pong_values[output_index], value_format);
                    }

                    GpuConstantBufferOfType<DownsweepConstantBuffer> downsweep_cb(gpu_system_, "downsweep_cb");
                    downsweep_cb->num_items = num_items;
                    downsweep_cb->pass = pass;
                    downsweep_cb->base_shift = pass_begin_bit;
                    downsweep_cb->bits = pass_end_bit - pass_begin_bit;
                    downsweep_cb.UploadStaging();

                    const GpuConstantBufferView downsweep_cbv(gpu_system_, downsweep_cb);

                    std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                        {"param_cb", &downsweep_cbv},
                    };
                    std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                        {"keys", &keys_srv},
                        {"values", &values_srv},
                        {"global_hist", &global_histogram_srv},
                        {"partition_hist", &partition_histogram_srv},
                    };
                    std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                        {"output_keys", &output_keys_uav},
                        {"output_values", &output_values_uav},
                    };
                    const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
                    cmd_list.Compute(*downsweep_pipeline, num_partitions, 1, 1, shader_binding);
                }
            }
        }

    private:
        GpuSystem& gpu_system_;

        struct UpsweepConstantBuffer
        {
            uint32_t num_items;
            uint32_t pass;
            uint32_t base_shift;
            uint32_t bits;
        };
        GpuComputePipeline upsweep_uint2_uint_pipeline_;
        GpuComputePipeline upsweep_uint_uint_pipeline_;

        struct ScanConstantBuffer
        {
            uint32_t num_items;
            uint32_t pass;
            uint32_t padding[2];
        };
        GpuComputePipeline scan_uint2_uint_pipeline_;
        GpuComputePipeline scan_uint_uint_pipeline_;

        struct DownsweepConstantBuffer
        {
            uint32_t num_items;
            uint32_t pass;
            uint32_t base_shift;
            uint32_t bits;
        };
        GpuComputePipeline downsweep_uint2_uint_pipeline_;
        GpuComputePipeline downsweep_uint_uint_pipeline_;
    };

    Sorter::Sorter() noexcept = default;
    Sorter::Sorter(AIHoloImagerInternal& aihi) : impl_(std::make_unique<Impl>(aihi))
    {
    }

    Sorter::~Sorter() noexcept = default;

    Sorter::Sorter(Sorter&& other) noexcept = default;
    Sorter& Sorter::operator=(Sorter&& other) noexcept = default;

    void Sorter::RadixSort(GpuCommandList& cmd_list, const GpuBuffer& keys, GpuFormat key_format, const GpuBuffer& values,
        GpuFormat value_format, uint32_t num_items, GpuBuffer& sorted_keys, GpuBuffer& sorted_values, uint32_t bits)
    {
        impl_->RadixSort(cmd_list, keys, key_format, values, value_format, num_items, sorted_keys, sorted_values, bits);
    }
} // namespace AIHoloImager
