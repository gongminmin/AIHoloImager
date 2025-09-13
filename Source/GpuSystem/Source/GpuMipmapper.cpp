// Copyright (c) 2025 Minmin Gong
//

#include "Gpu/GpuMipmapper.hpp"

#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuConstantBuffer.hpp"
#include "Gpu/GpuResourceViews.hpp"
#include "Gpu/GpuShader.hpp"
#include "Gpu/GpuSystem.hpp"

#include "CompiledShader/GenMipmapCs.h"

namespace AIHoloImager
{
    class GpuMipmapper::Impl
    {
    public:
        explicit Impl(GpuSystem& gpu_system) : gpu_system_(gpu_system)
        {
            const ShaderInfo shader = {GenMipmapCs_shader, 1, 1, 1, 1};
            gen_mipmap_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
        }

        void Generate(GpuCommandList& cmd_list, GpuTexture2D& texture, GpuSampler::Filter filter)
        {
            const uint32_t num_levels = texture.MipLevels();
            if (num_levels == 1)
            {
                return;
            }

            constexpr uint32_t BlockDim = 16;

            const GpuDynamicSampler sampler(
                gpu_system_, GpuSampler::Filters(filter, filter), GpuSampler::AddressModes(GpuDynamicSampler::AddressMode::Clamp));

            for (uint32_t i = 1; i < num_levels; ++i)
            {
                const uint32_t this_width = texture.Width(i);
                const uint32_t this_height = texture.Height(i);

                GpuConstantBufferOfType<GenMipmapConstantBuffer> gen_mipmap_cb(gpu_system_, L"gen_mipmap_cb");
                gen_mipmap_cb->this_level_width = this_width;
                gen_mipmap_cb->this_level_height = this_height;
                gen_mipmap_cb.UploadStaging();

                const GpuShaderResourceView last_level_srv(gpu_system_, texture, i - 1);
                GpuUnorderedAccessView this_level_uav(gpu_system_, texture, i);

                const GpuConstantBuffer* cbs[] = {&gen_mipmap_cb};
                const GpuShaderResourceView* srvs[] = {&last_level_srv};
                GpuUnorderedAccessView* uavs[] = {&this_level_uav};
                const GpuDynamicSampler* samplers[] = {&sampler};
                const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs, samplers};
                cmd_list.Compute(gen_mipmap_pipeline_, DivUp(this_width, BlockDim), DivUp(this_height, BlockDim), 1, shader_binding);
            }
        }

    private:
        GpuSystem& gpu_system_;

        struct GenMipmapConstantBuffer
        {
            uint32_t this_level_width;
            uint32_t this_level_height;
            uint32_t padding[2];
        };
        GpuComputePipeline gen_mipmap_pipeline_;
    };


    GpuMipmapper::GpuMipmapper() = default;
    GpuMipmapper::GpuMipmapper(GpuSystem& gpu_system) : impl_(std::make_unique<Impl>(gpu_system))
    {
    }

    GpuMipmapper::~GpuMipmapper() noexcept = default;

    GpuMipmapper::GpuMipmapper(GpuMipmapper&& other) noexcept = default;
    GpuMipmapper& GpuMipmapper::operator=(GpuMipmapper&& other) noexcept = default;

    void GpuMipmapper::Generate(GpuCommandList& cmd_list, GpuTexture2D& texture, GpuSampler::Filter filter)
    {
        impl_->Generate(cmd_list, texture, filter);
    }
} // namespace AIHoloImager
