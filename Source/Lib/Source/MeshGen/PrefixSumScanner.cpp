// Copyright (c) 2024 Minmin Gong
//

#include "PrefixSumScanner.hpp"

#include "Gpu/GpuBufferHelper.hpp"
#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuResourceViews.hpp"
#include "Gpu/GpuShader.hpp"

#include "CompiledShader/ApplySumUint2Cs.h"
#include "CompiledShader/ApplySumUintCs.h"
#include "CompiledShader/PrefixSumUint2Cs.h"
#include "CompiledShader/PrefixSumUintCs.h"

namespace AIHoloImager
{
    class PrefixSumScanner::Impl
    {
    public:
        Impl(GpuSystem& gpu_system) : gpu_system_(gpu_system)
        {
            {
                const ShaderInfo shader = {PrefixSumUintCs_shader, 1, 1, 2};
                prefix_sum_uint_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }
            {
                const ShaderInfo shader = {PrefixSumUint2Cs_shader, 1, 1, 2};
                prefix_sum_uint2_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }
            {
                const ShaderInfo shader = {ApplySumUintCs_shader, 1, 1, 1};
                apply_sum_uint_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }
            {
                const ShaderInfo shader = {ApplySumUint2Cs_shader, 1, 1, 1};
                apply_sum_uint2_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }
        }

        void Scan(GpuCommandList& cmd_list, const GpuBuffer& input, GpuBuffer& output, uint32_t num_elems, DXGI_FORMAT format)
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

                scan_scanned_buffers.push_back(GpuBuffer(gpu_system_, buff_size * elem_size, D3D12_HEAP_TYPE_DEFAULT,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON, std::format(L"scan_sum_buffer_{}", i)));
            }

            GpuComputePipeline* prefix_sum_pipeline;
            GpuComputePipeline* apply_sum_pipeline;
            switch (format)
            {
            case DXGI_FORMAT_R32_UINT:
                prefix_sum_pipeline = &prefix_sum_uint_pipeline_;
                apply_sum_pipeline = &apply_sum_uint_pipeline_;
                break;
            case DXGI_FORMAT_R32G32_UINT:
                prefix_sum_pipeline = &prefix_sum_uint2_pipeline_;
                apply_sum_pipeline = &apply_sum_uint2_pipeline_;
                break;

            default:
                Unreachable();
            }

            for (size_t i = 0; i < buff_sizes.size(); ++i)
            {
                ConstantBuffer<PrefixSumConstantBuffer> prefix_sum_cb(gpu_system_, 1, L"prefix_sum_cb");
                prefix_sum_cb->size = buff_sizes[i];
                prefix_sum_cb->from_input = (i == 0);
                prefix_sum_cb.UploadToGpu();
                buff_size = DivUp(buff_size, BlockDim);

                GpuShaderResourceView input_srv(gpu_system_, input, format);
                GpuUnorderedAccessView input_output_uav(gpu_system_, i == 0 ? output : scan_scanned_buffers[i - 1], format);
                GpuUnorderedAccessView sum_output_uav(gpu_system_, scan_scanned_buffers[i], format);

                const GeneralConstantBuffer* cbs[] = {&prefix_sum_cb};
                const GpuShaderResourceView* srvs[] = {&input_srv};
                GpuUnorderedAccessView* uavs[] = {&input_output_uav, &sum_output_uav};
                const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
                cmd_list.Compute(*prefix_sum_pipeline, DivUp(buff_sizes[i], BlockDim), 1, 1, shader_binding);
            }

            for (int i = static_cast<int>(scan_scanned_buffers.size() - 2); i >= 0; --i)
            {
                ConstantBuffer<ApplySumConstantBuffer> apply_sum_cb(gpu_system_, 1, L"apply_sum_cb");
                apply_sum_cb->size = buff_sizes[i];
                apply_sum_cb.UploadToGpu();

                GpuShaderResourceView sum_srv(gpu_system_, scan_scanned_buffers[i], format);
                GpuUnorderedAccessView output_uav(gpu_system_, i == 0 ? output : scan_scanned_buffers[i - 1], format);

                if (i == 0)
                {
                    cmd_list.Copy(output, num_elems * elem_size, scan_scanned_buffers.back(), 0, elem_size);
                }

                const GeneralConstantBuffer* cbs[] = {&apply_sum_cb};
                const GpuShaderResourceView* srvs[] = {&sum_srv};
                GpuUnorderedAccessView* uavs[] = {&output_uav};
                const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
                cmd_list.Compute(*apply_sum_pipeline, DivUp(buff_sizes[i], BlockDim), 1, 1, shader_binding);
            }
        }

    private:
        GpuSystem& gpu_system_;

        struct PrefixSumConstantBuffer
        {
            uint32_t size;
            uint32_t from_input;
            uint32_t padding[2];
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

    PrefixSumScanner::PrefixSumScanner(GpuSystem& gpu_system) : impl_(std::make_unique<Impl>(gpu_system))
    {
    }

    PrefixSumScanner::~PrefixSumScanner() noexcept = default;

    PrefixSumScanner::PrefixSumScanner(PrefixSumScanner&& other) noexcept = default;
    PrefixSumScanner& PrefixSumScanner::operator=(PrefixSumScanner&& other) noexcept = default;

    void PrefixSumScanner::Scan(GpuCommandList& cmd_list, const GpuBuffer& input, GpuBuffer& output, uint32_t num_elems, DXGI_FORMAT format)
    {
        impl_->Scan(cmd_list, input, output, num_elems, format);
    }
} // namespace AIHoloImager
