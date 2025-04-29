// Copyright (c) 2025 Minmin Gong
//

#include <tuple>

#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4100) // Ignore unreferenced formal parameters
    #pragma warning(disable : 4127) // Ignore constant conditional expression
    #pragma warning(disable : 4244) // Ignore type conversion from `int` to `float`
    #pragma warning(disable : 4251) // Ignore non dll-interface as member
    #pragma warning(disable : 4267) // Ignore type conversion from `size_t` to something else
    #pragma warning(disable : 4324) // Ignore padded structure
    #pragma warning(disable : 4275) // Ignore non dll-interface base class
#endif
#include <torch/types.h>
#ifdef _MSC_VER
    #pragma warning(pop)
#endif

#include "../GpuDiffRender.hpp"
#include "Gpu/GpuBuffer.hpp"
#include "Gpu/GpuTexture.hpp"
#include "MiniCudaRt.hpp"

namespace AIHoloImager
{
    class GpuDiffRenderTorch
    {
    public:
        GpuDiffRenderTorch(size_t gpu_system, torch::Device torch_device);
        ~GpuDiffRenderTorch();

        torch::Tensor Rasterize(torch::Tensor positions, torch::Tensor indices, std::tuple<uint32_t, uint32_t> resolution);

        torch::Tensor Interpolate(torch::Tensor vtx_attribs, torch::Tensor gbuffer, torch::Tensor indices);

        struct AntiAliasOppositeVertices
        {
            GpuBuffer opposite_vertices;
        };

        AntiAliasOppositeVertices AntiAliasConstructOppositeVertices(torch::Tensor indices);

        torch::Tensor AntiAlias(torch::Tensor shading, torch::Tensor gbuffer, torch::Tensor positions, torch::Tensor indices,
            const AntiAliasOppositeVertices* opposite_vertices = nullptr);

    private:
        torch::Tensor RasterizeFwd(torch::Tensor positions, torch::Tensor indices, std::tuple<uint32_t, uint32_t> resolution);
        torch::Tensor RasterizeBwd(torch::Tensor grad_gbuffer);

        torch::Tensor InterpolateFwd(torch::Tensor vtx_attribs, torch::Tensor gbuffer, torch::Tensor indices);
        std::tuple<torch::Tensor, torch::Tensor> InterpolateBwd(torch::Tensor grad_shading);

        torch::Tensor AntiAliasFwd(torch::Tensor shading, torch::Tensor gbuffer, torch::Tensor positions, torch::Tensor indices,
            const AntiAliasOppositeVertices* opposite_vertices = nullptr);
        std::tuple<torch::Tensor, torch::Tensor> AntiAliasBwd(torch::Tensor grad_anti_aliased);

        void Convert(
            GpuCommandList& cmd_list, torch::Tensor tensor, GpuBuffer& buff, GpuHeap heap, GpuResourceFlag flags, std::wstring_view name);
        void Convert(GpuCommandList& cmd_list, torch::Tensor tensor, GpuTexture2D& tex, GpuFormat format, GpuResourceFlag flags,
            std::wstring_view name);
        torch::Tensor Convert(GpuCommandList& cmd_list, const GpuBuffer& buff, const torch::IntArrayRef& size, torch::Dtype data_type);
        torch::Tensor Convert(GpuCommandList& cmd_list, const GpuTexture2D& tex);

        MiniCudaRt::ExternalMemory_t ImportFromResource(const GpuResource& resource);
        void WaitExternalSemaphore(uint64_t fence_val);
        void SignalExternalSemaphore(uint64_t fence_val);
        MiniCudaRt::ChannelFormatDesc FormatDesc(GpuFormat format);

    private:
        GpuSystem& gpu_system_;
        torch::Device torch_device_;

        GpuDiffRender gpu_dr_;

        MiniCudaRt cuda_rt_;

        bool uses_cuda_copy_;
        MiniCudaRt::ExternalSemaphore_t ext_semaphore_{};
        MiniCudaRt::Stream_t copy_stream_{};

        struct RasterizeIntermediate
        {
            GpuBuffer positions;
            GpuBuffer indices;
            GpuTexture2D gbuffer;

            GpuTexture2D grad_gbuffer;
            GpuBuffer grad_positions;
        };
        RasterizeIntermediate rast_intermediate_;

        struct InterpolateIntermediate
        {
            GpuBuffer vtx_attribs;
            uint32_t num_attribs;
            GpuTexture2D gbuffer;
            GpuBuffer indices;

            GpuBuffer shading;
            GpuBuffer grad_shading;
            GpuBuffer grad_vtx_attribs;
            GpuTexture2D grad_gbuffer;
        };
        InterpolateIntermediate interpolate_intermediate_;

        struct AntiAliasIntermediate
        {
            GpuBuffer shading;
            GpuTexture2D gbuffer;
            GpuBuffer positions;
            GpuBuffer indices;

            GpuBuffer anti_aliased;
            GpuBuffer grad_anti_aliased;
            GpuBuffer grad_shading;
            GpuBuffer grad_positions;
        };
        AntiAliasIntermediate aa_intermediate_;
    };
} // namespace AIHoloImager
