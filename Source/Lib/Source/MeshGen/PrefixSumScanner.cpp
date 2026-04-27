// Copyright (c) 2024-2026 Minmin Gong
//

#include "PrefixSumScanner.hpp"

#include "Base/ErrorHandling.hpp"
#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuResourceViews.hpp"
#include "Gpu/GpuShader.hpp"

#include "CompiledShader/MeshGen/PrefixSumScanner/ApplySumUint2Cs.h"
#include "CompiledShader/MeshGen/PrefixSumScanner/ApplySumUintCs.h"
#include "CompiledShader/MeshGen/PrefixSumScanner/PrefixSumUint2Cs.h"
#include "CompiledShader/MeshGen/PrefixSumScanner/PrefixSumUintCs.h"

namespace AIHoloImager
{
    class PrefixSumScanner::Impl
    {
    public:
        explicit Impl(AIHoloImagerInternal& aihi) : gpu_system_(aihi.GpuSystemInstance())
        {
            {
                const ShaderInfo shader = {DEFINE_SHADER(PrefixSumUintCs)};
                prefix_sum_uint_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }
            {
                const ShaderInfo shader = {DEFINE_SHADER(PrefixSumUint2Cs)};
                prefix_sum_uint2_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }
            {
                const ShaderInfo shader = {DEFINE_SHADER(ApplySumUintCs)};
                apply_sum_uint_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }
            {
                const ShaderInfo shader = {DEFINE_SHADER(ApplySumUint2Cs)};
                apply_sum_uint2_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }
        }

        void Scan(GpuCommandList& cmd_list, const GpuBuffer& input, GpuBuffer& output, uint32_t num_elems, GpuFormat format, bool exclusive)
        {
            constexpr uint32_t BlockDim = 256;

            const uint32_t elem_size = FormatSize(format);

            uint32_t buff_size = num_elems;
            std::vector<uint32_t> buff_sizes;
            std::vector<GpuBuffer> scan_scanned_buffers;
            for (uint32_t i = 0; buff_size != 1; ++i)
            {
                buff_sizes.push_back(buff_size);
                buff_size = DivUp(buff_size, BlockDim);

                scan_scanned_buffers.push_back(GpuBuffer(gpu_system_, buff_size * elem_size, GpuHeap::Default,
                    GpuResourceFlag::UnorderedAccess, std::format("PrefixSumScanner.scan_sum_buffer_{}", i)));
            }

            GpuComputePipeline* prefix_sum_pipeline;
            GpuComputePipeline* apply_sum_pipeline;
            switch (format)
            {
            case GpuFormat::R32_Uint:
                prefix_sum_pipeline = &prefix_sum_uint_pipeline_;
                apply_sum_pipeline = &apply_sum_uint_pipeline_;
                break;
            case GpuFormat::RG32_Uint:
                prefix_sum_pipeline = &prefix_sum_uint_pipeline_;
                apply_sum_pipeline = &apply_sum_uint2_pipeline_;
                break;

            default:
                Unreachable();
            }

            for (size_t i = 0; i < buff_sizes.size(); ++i)
            {
                GpuConstantBufferOfType<PrefixSumConstantBuffer> prefix_sum_cb(gpu_system_, "prefix_sum_cb");
                prefix_sum_cb->size = buff_sizes[i];
                prefix_sum_cb->from_input = (i == 0);
                prefix_sum_cb->exclusive = exclusive || (i != 0);
                prefix_sum_cb.UploadStaging();
                buff_size = DivUp(buff_size, BlockDim);

                const GpuConstantBufferView prefix_sum_cbv(gpu_system_, prefix_sum_cb);
                const GpuShaderResourceView input_srv(gpu_system_, input, format);
                GpuUnorderedAccessView input_output_uav(gpu_system_, i == 0 ? output : scan_scanned_buffers[i - 1], format);
                GpuUnorderedAccessView sum_output_uav(gpu_system_, scan_scanned_buffers[i], format);

                std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                    {"param_cb", &prefix_sum_cbv},
                };
                std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                    {"input", &input_srv},
                };
                std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                    {"input_output", &input_output_uav},
                    {"sum_output", &sum_output_uav},
                };
                const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
                cmd_list.Compute(*prefix_sum_pipeline, {DivUp(buff_sizes[i], BlockDim), 1, 1}, shader_binding);
            }

            for (int i = static_cast<int>(scan_scanned_buffers.size() - 2); i >= 0; --i)
            {
                GpuConstantBufferOfType<ApplySumConstantBuffer> apply_sum_cb(gpu_system_, "apply_sum_cb");
                apply_sum_cb->size = buff_sizes[i];
                apply_sum_cb.UploadStaging();

                const GpuConstantBufferView apply_sum_cbv(gpu_system_, apply_sum_cb);
                const GpuShaderResourceView sum_srv(gpu_system_, scan_scanned_buffers[i], format);
                GpuUnorderedAccessView output_uav(gpu_system_, i == 0 ? output : scan_scanned_buffers[i - 1], format);

                const uint32_t sum_offset = num_elems * elem_size;
                if (exclusive && (i == 0) && (output.Size() > sum_offset))
                {
                    cmd_list.Copy(output, sum_offset, scan_scanned_buffers.back(), 0, elem_size);
                }

                std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                    {"param_cb", &apply_sum_cbv},
                };
                std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                    {"sum", &sum_srv},
                };
                std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                    {"output", &output_uav},
                };

                const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
                cmd_list.Compute(*apply_sum_pipeline, {DivUp(buff_sizes[i], BlockDim), 1, 1}, shader_binding);
            }
        }

    private:
        GpuSystem& gpu_system_;

        struct PrefixSumConstantBuffer
        {
            uint32_t size;
            uint32_t from_input;
            uint32_t exclusive;
            uint32_t padding;
        };
        GpuComputePipeline prefix_sum_uint_pipeline_;
        GpuComputePipeline prefix_sum_uint2_pipeline_;

        struct ApplySumConstantBuffer
        {
            uint32_t size;
            uint32_t padding[3];
        };
        GpuComputePipeline apply_sum_uint_pipeline_;
        GpuComputePipeline apply_sum_uint2_pipeline_;
    };

    PrefixSumScanner::PrefixSumScanner() noexcept = default;
    PrefixSumScanner::PrefixSumScanner(AIHoloImagerInternal& aihi) : impl_(std::make_unique<Impl>(aihi))
    {
    }

    PrefixSumScanner::~PrefixSumScanner() noexcept = default;

    PrefixSumScanner::PrefixSumScanner(PrefixSumScanner&& other) noexcept = default;
    PrefixSumScanner& PrefixSumScanner::operator=(PrefixSumScanner&& other) noexcept = default;

    void PrefixSumScanner::Scan(
        GpuCommandList& cmd_list, const GpuBuffer& input, GpuBuffer& output, uint32_t num_elems, GpuFormat format, bool exclusive)
    {
        impl_->Scan(cmd_list, input, output, num_elems, format, exclusive);
    }
} // namespace AIHoloImager
