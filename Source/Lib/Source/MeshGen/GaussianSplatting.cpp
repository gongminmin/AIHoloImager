// Copyright (c) 2026 Minmin Gong
//

// An implementation of "3D Gaussian Splatting for Real-Time Rendering of Radiance Fields". The algorithm and code structure follows
// https://github.com/autonomousvision/mip-splatting/tree/main/submodules/diff-gaussian-rasterization, which is a derivant of
// https://github.com/graphdeco-inria/diff-gaussian-rasterization.

#include "GaussianSplatting.hpp"

#include <bit>
#ifdef AIHI_KEEP_INTERMEDIATES
    #include <fstream>
#endif

#include "AIHoloImager/Mesh.hpp"
#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuSystem.hpp"
#include "Util/PerfProfiler.hpp"

#include "PrefixSumScanner.hpp"
#include "Sorter.hpp"

#include "CompiledShader/MeshGen/GaussianSplatting/DupWithKeysCs.h"
#include "CompiledShader/MeshGen/GaussianSplatting/IdentifyTileRangesCs.h"
#include "CompiledShader/MeshGen/GaussianSplatting/PreprocessCs.h"
#include "CompiledShader/MeshGen/GaussianSplatting/RenderCs.h"

namespace
{
    constexpr uint32_t NumCoefficientsFromShDegrees(uint32_t degrees) noexcept
    {
        return (degrees + 1) * (degrees + 1);
    }
} // namespace

namespace AIHoloImager
{
    class GaussianSplatting::Impl
    {
    public:
        explicit Impl(AIHoloImagerInternal& aihi) : aihi_(aihi), prefix_sum_scanner_(aihi), sorter_(aihi)
        {
            auto& gpu_system = aihi_.GpuSystemInstance();

            {
                const ShaderInfo shader = {DEFINE_SHADER(PreprocessCs)};
                preprocess_pipeline_ = GpuComputePipeline(gpu_system, shader, {});
            }
            {
                const ShaderInfo shader = {DEFINE_SHADER(DupWithKeysCs)};
                dup_with_keys_pipeline_ = GpuComputePipeline(gpu_system, shader, {});
            }
            {
                const ShaderInfo shader = {DEFINE_SHADER(IdentifyTileRangesCs)};
                identify_tile_ranges_pipeline_ = GpuComputePipeline(gpu_system, shader, {});
            }
            {
                const ShaderInfo shader = {DEFINE_SHADER(RenderCs)};
                render_pipeline_ = GpuComputePipeline(gpu_system, shader, {});
            }
        }

        void Render(const Gaussians& gaussians, const glm::mat4x4& view_mtx, const glm::mat4x4& proj_mtx, float kernel_size,
            GpuTexture2D& rendered_image)
        {
            auto& gpu_system = aihi_.GpuSystemInstance();

            constexpr uint32_t ImgTileX = 16;
            constexpr uint32_t ImgTileY = 16;

            const uint32_t width = rendered_image.Width(0);
            const uint32_t height = rendered_image.Height(0);
            const float tan_fov_x = 1 / proj_mtx[0][0];
            const float tan_fov_y = 1 / proj_mtx[1][1];
            const float focal_x = width * proj_mtx[0][0] / 2;
            const float focal_y = height * proj_mtx[1][1] / 2;

            const glm::uvec2 tile_grid(DivUp(width, ImgTileX), DivUp(height, ImgTileY));

            auto cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Compute);

            PerfRegion gsplat_perf(aihi_.PerfProfilerInstance(), "GaussianSplatting Render", &cmd_list);

            struct GeometryState
            {
                GpuBuffer depth;
                GpuBuffer screen_pos;
                GpuBuffer conic_opacity;
                GpuBuffer color;
                GpuBuffer point_offset;
                GpuBuffer radius;
                GpuBuffer tiles_touched;
                GpuBuffer num_rendered;
            };

            GeometryState geom_state;
            geom_state.depth = GpuBuffer(gpu_system, gaussians.num_gaussians * sizeof(float), GpuHeap::Default,
                GpuResourceFlag::UnorderedAccess, "gsplat.geom_state.depth");
            geom_state.screen_pos = GpuBuffer(gpu_system, gaussians.num_gaussians * sizeof(glm::vec2), GpuHeap::Default,
                GpuResourceFlag::UnorderedAccess, "gsplat.geom_state.screen_pos");
            geom_state.conic_opacity = GpuBuffer(gpu_system, gaussians.num_gaussians * sizeof(glm::vec4), GpuHeap::Default,
                GpuResourceFlag::UnorderedAccess, "gsplat.geom_state.conic_opacity");
            geom_state.color = GpuBuffer(gpu_system, gaussians.num_gaussians * sizeof(glm::vec3), GpuHeap::Default,
                GpuResourceFlag::UnorderedAccess, "gsplat.geom_state.color");
            geom_state.point_offset = GpuBuffer(gpu_system, gaussians.num_gaussians * sizeof(uint32_t), GpuHeap::Default,
                GpuResourceFlag::UnorderedAccess, "gsplat.geom_state.point_offset");
            geom_state.radius = GpuBuffer(gpu_system, gaussians.num_gaussians * sizeof(uint32_t), GpuHeap::Default,
                GpuResourceFlag::UnorderedAccess, "gsplat.geom_state.radius");
            geom_state.tiles_touched = GpuBuffer(gpu_system, gaussians.num_gaussians * sizeof(uint32_t), GpuHeap::Default,
                GpuResourceFlag::UnorderedAccess, "gsplat.geom_state.tiles_touched");
            geom_state.num_rendered =
                GpuBuffer(gpu_system, sizeof(uint32_t), GpuHeap::Default, GpuResourceFlag::None, "geom_state.num_rendered");

            {
                constexpr uint32_t BlockDim = 256;

                GpuConstantBufferOfType<PreprocessConstantBuffer> preprocess_cb(gpu_system, "preprocess_cb");
                preprocess_cb->num_gaussians = gaussians.num_gaussians;
                preprocess_cb->sh_degrees = gaussians.sh_degrees;
                preprocess_cb->num_coeffs = NumCoefficientsFromShDegrees(gaussians.sh_degrees);
                preprocess_cb->kernel_size = kernel_size;
                preprocess_cb->view_mtx = glm::transpose(view_mtx);
                preprocess_cb->proj_mtx = glm::transpose(proj_mtx);
                preprocess_cb->focal = glm::vec2(focal_x, focal_y);
                preprocess_cb->tan_fov = glm::vec2(tan_fov_x, tan_fov_y);
                preprocess_cb->width_height = glm::uvec2(width, height);
                preprocess_cb->tile_grid = tile_grid;
                preprocess_cb.UploadStaging();
                const GpuConstantBufferView preprocess_cbv(gpu_system, preprocess_cb);

                const GpuShaderResourceView pos_srv(gpu_system, gaussians.positions, GpuFormat::RGB32_Float);
                const GpuShaderResourceView scale_srv(gpu_system, gaussians.scales, GpuFormat::RGB32_Float);
                const GpuShaderResourceView rotation_srv(gpu_system, gaussians.rotations, GpuFormat::RGBA32_Float);
                const GpuShaderResourceView sh_srv(gpu_system, gaussians.shs, GpuFormat::RGB32_Float);
                const GpuShaderResourceView opacity_srv(gpu_system, gaussians.opacities, GpuFormat::R32_Float);

                GpuUnorderedAccessView radius_buff_uav(gpu_system, geom_state.radius, GpuFormat::R32_Uint);
                GpuUnorderedAccessView tiles_touched_uav(gpu_system, geom_state.tiles_touched, GpuFormat::R32_Uint);
                GpuUnorderedAccessView color_uav(gpu_system, geom_state.color, GpuFormat::R32_Float);
                GpuUnorderedAccessView conic_opacity_uav(gpu_system, geom_state.conic_opacity, GpuFormat::RGBA32_Float);
                GpuUnorderedAccessView depth_uav(gpu_system, geom_state.depth, GpuFormat::R32_Float);
                GpuUnorderedAccessView screen_pos_uav(gpu_system, geom_state.screen_pos, GpuFormat::RG32_Float);

                std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                    {"param_cb", &preprocess_cbv},
                };
                std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                    {"pos_buff", &pos_srv},
                    {"scale_buff", &scale_srv},
                    {"rotation_buff", &rotation_srv},
                    {"sh_buff", &sh_srv},
                    {"opacity_buff", &opacity_srv},
                };
                std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                    {"radius_buff", &radius_buff_uav},
                    {"tiles_touched_buff", &tiles_touched_uav},
                    {"color_buff", &color_uav},
                    {"conic_opacity_buff", &conic_opacity_uav},
                    {"depth_buff", &depth_uav},
                    {"screen_pos_buff", &screen_pos_uav},
                };
                const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
                cmd_list.Compute(preprocess_pipeline_, DivUp(gaussians.num_gaussians, BlockDim), 1, 1, shader_binding);
            }

            uint32_t num_rendered;
            {
                prefix_sum_scanner_.Scan(
                    cmd_list, geom_state.tiles_touched, geom_state.point_offset, gaussians.num_gaussians, GpuFormat::R32_Uint, false);

                cmd_list.Copy(geom_state.num_rendered, 0, geom_state.point_offset, (gaussians.num_gaussians - 1) * sizeof(uint32_t),
                    sizeof(uint32_t));

                auto rb_future = cmd_list.ReadBackAsync(geom_state.num_rendered, &num_rendered, sizeof(num_rendered));

                gpu_system.ExecuteAndReset(cmd_list);

                rb_future.wait();

                geom_state.tiles_touched = GpuBuffer();
                geom_state.num_rendered = GpuBuffer();
            }

            struct BinningState
            {
                GpuBuffer point_keys;
                GpuBuffer point_ids;
            };

            BinningState binning_state;
            binning_state.point_keys = GpuBuffer(gpu_system, num_rendered * sizeof(uint64_t), GpuHeap::Default,
                GpuResourceFlag::UnorderedAccess, "gsplat.binning_state.point_keys");
            binning_state.point_ids = GpuBuffer(gpu_system, num_rendered * sizeof(uint32_t), GpuHeap::Default,
                GpuResourceFlag::UnorderedAccess, "gsplat.binning_state.point_ids");

            {
                constexpr uint32_t BlockDim = 256;

                GpuConstantBufferOfType<DupWithKeysConstantBuffer> dup_with_keys_cb(gpu_system, "dup_with_keys_cb");
                dup_with_keys_cb->num_gaussians = gaussians.num_gaussians;
                dup_with_keys_cb->tile_grid = tile_grid;
                dup_with_keys_cb.UploadStaging();
                const GpuConstantBufferView dup_with_keys_cbv(gpu_system, dup_with_keys_cb);

                const GpuShaderResourceView screen_pos_srv(gpu_system, geom_state.screen_pos, GpuFormat::RG32_Float);
                const GpuShaderResourceView depth_srv(gpu_system, geom_state.depth, GpuFormat::R32_Float);
                const GpuShaderResourceView point_offset_srv(gpu_system, geom_state.point_offset, GpuFormat::R32_Uint);
                const GpuShaderResourceView radius_buff_srv(gpu_system, geom_state.radius, GpuFormat::R32_Uint);

                GpuUnorderedAccessView point_keys_unsorted_uav(gpu_system, binning_state.point_keys, GpuFormat::RG32_Uint);
                GpuUnorderedAccessView point_ids_unsorted_uav(gpu_system, binning_state.point_ids, GpuFormat::R32_Uint);

                std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                    {"param_cb", &dup_with_keys_cbv},
                };
                std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                    {"screen_pos_buff", &screen_pos_srv},
                    {"depth_buff", &depth_srv},
                    {"point_offset_buff", &point_offset_srv},
                    {"radius_buff", &radius_buff_srv},
                };
                std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                    {"point_keys_unsorted_buff", &point_keys_unsorted_uav},
                    {"point_ids_unsorted_buff", &point_ids_unsorted_uav},
                };
                const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
                cmd_list.Compute(dup_with_keys_pipeline_, DivUp(gaussians.num_gaussians, BlockDim), 1, 1, shader_binding);

                geom_state.depth = GpuBuffer();
                geom_state.point_offset = GpuBuffer();
                geom_state.radius = GpuBuffer();
            }

            {
                const uint32_t bits = 32 - std::countl_zero(tile_grid.x * tile_grid.y);
                sorter_.RadixSort(cmd_list, binning_state.point_keys, GpuFormat::RG32_Uint, binning_state.point_ids, GpuFormat::R32_Uint,
                    num_rendered, binning_state.point_keys, binning_state.point_ids, 32 + bits);
            }

            struct ImageState
            {
                GpuBuffer ranges;
                GpuUnorderedAccessView ranges_uav;
            };

            ImageState img_state;
            img_state.ranges = GpuBuffer(gpu_system, tile_grid.x * tile_grid.y * sizeof(glm::uvec2), GpuHeap::Default,
                GpuResourceFlag::UnorderedAccess, "gsplat.img_state.ranges");
            img_state.ranges_uav = GpuUnorderedAccessView(gpu_system, img_state.ranges, GpuFormat::R32_Uint);

            {
                const uint32_t clear_clr[] = {0, 0, 0, 0};
                cmd_list.Clear(img_state.ranges_uav, clear_clr);
            }

            if (num_rendered > 0)
            {
                constexpr uint32_t BlockDim = 256;

                GpuConstantBufferOfType<IdentifyTileRangesConstantBuffer> identify_tile_ranges_cb(gpu_system, "identify_tile_ranges_cb");
                identify_tile_ranges_cb->length = num_rendered;
                identify_tile_ranges_cb.UploadStaging();
                const GpuConstantBufferView identify_tile_ranges_cbv(gpu_system, identify_tile_ranges_cb);

                const GpuShaderResourceView point_keys_srv(gpu_system, binning_state.point_keys, GpuFormat::RG32_Uint);

                std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                    {"param_cb", &identify_tile_ranges_cbv},
                };
                std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                    {"point_keys_buff", &point_keys_srv},
                };
                std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                    {"ranges_buff", &img_state.ranges_uav},
                };
                const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
                cmd_list.Compute(identify_tile_ranges_pipeline_, DivUp(num_rendered, BlockDim), 1, 1, shader_binding);
            }

            {
                GpuConstantBufferOfType<RenderConstantBuffer> render_cb(gpu_system, "render_cb");
                render_cb->width_height = glm::uvec2(width, height);
                render_cb.UploadStaging();
                const GpuConstantBufferView render_cbv(gpu_system, render_cb);

                const GpuShaderResourceView ranges_srv(gpu_system, img_state.ranges, GpuFormat::RG32_Uint);
                const GpuShaderResourceView point_ids_srv(gpu_system, binning_state.point_ids, GpuFormat::R32_Uint);
                const GpuShaderResourceView screen_pos_srv(gpu_system, geom_state.screen_pos, GpuFormat::RG32_Float);
                const GpuShaderResourceView point_colors_srv(gpu_system, geom_state.color,
                    GpuFormat::R32_Float); // Vulkan doesn't support a buffer with RGB32_Float of both SRV and UAV
                const GpuShaderResourceView conic_opacity_srv(gpu_system, geom_state.conic_opacity, GpuFormat::RGBA32_Float);

                GpuUnorderedAccessView rendered_image_uav(gpu_system, rendered_image, 0);

                std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                    {"param_cb", &render_cbv},
                };
                std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                    {"ranges_buff", &ranges_srv},
                    {"point_ids_buff", &point_ids_srv},
                    {"screen_pos_buff", &screen_pos_srv},
                    {"point_colors_buff", &point_colors_srv},
                    {"conic_opacity_buff", &conic_opacity_srv},
                };
                std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                    {"rendered_image", &rendered_image_uav},
                };
                const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
                cmd_list.Compute(render_pipeline_, DivUp(width, ImgTileX), DivUp(height, ImgTileY), 1, shader_binding);
            }

#ifdef AIHI_KEEP_INTERMEDIATES
            rendered_image.Transition(cmd_list, GpuResourceState::Common);
#endif

            gsplat_perf.End();
            gpu_system.Execute(std::move(cmd_list));
        }

    private:
        AIHoloImagerInternal& aihi_;

        struct PreprocessConstantBuffer
        {
            uint32_t num_gaussians;
            uint32_t sh_degrees;
            uint32_t num_coeffs;
            float kernel_size;

            glm::mat4x4 view_mtx;
            glm::mat4x4 proj_mtx;

            glm::vec2 focal;
            glm::vec2 tan_fov;

            glm::uvec2 width_height;
            glm::uvec2 tile_grid;
        };
        GpuComputePipeline preprocess_pipeline_;

        struct DupWithKeysConstantBuffer
        {
            uint32_t num_gaussians;
            glm::uvec2 tile_grid;
            uint32_t padding;
        };
        GpuComputePipeline dup_with_keys_pipeline_;

        struct IdentifyTileRangesConstantBuffer
        {
            uint32_t length;
            glm::uvec3 padding;
        };
        GpuComputePipeline identify_tile_ranges_pipeline_;

        struct RenderConstantBuffer
        {
            glm::uvec2 width_height;
            glm::uvec2 padding;
        };
        GpuComputePipeline render_pipeline_;

        PrefixSumScanner prefix_sum_scanner_;
        Sorter sorter_;
    };

    GaussianSplatting::GaussianSplatting() noexcept = default;
    GaussianSplatting::GaussianSplatting(AIHoloImagerInternal& aihi) : impl_(std::make_unique<Impl>(aihi))
    {
    }

    GaussianSplatting::~GaussianSplatting() noexcept = default;

    GaussianSplatting::GaussianSplatting(GaussianSplatting&& other) noexcept = default;
    GaussianSplatting& GaussianSplatting::operator=(GaussianSplatting&& other) noexcept = default;

    void GaussianSplatting::Render(const Gaussians& gaussians, const glm::mat4x4& view_mtx, const glm::mat4x4& proj_mtx, float kernel_size,
        GpuTexture2D& rendered_image)
    {
        impl_->Render(gaussians, view_mtx, proj_mtx, kernel_size, rendered_image);
    }

#ifdef AIHI_KEEP_INTERMEDIATES
    glm::vec3 Sh2Rgb(const glm::vec3* sh) noexcept
    {
        constexpr float C0 = 0.28209479177387814f;
        return sh[0] * C0 + 0.5f;
    }

    void SavePointCloud(GpuSystem& gpu_system, const Gaussians& gaussians, const std::filesystem::path& path)
    {
        const VertexAttrib pos_clr_vertex_attribs[] = {
            {VertexAttrib::Semantic::Position, 0, 3},
            {VertexAttrib::Semantic::Color, 0, 3},
        };
        constexpr uint32_t PosAttribIndex = 0;
        constexpr uint32_t ColorAttribIndex = 1;
        const VertexDesc pos_clr_vertex_desc(pos_clr_vertex_attribs);

        Mesh pc_mesh = Mesh(pos_clr_vertex_desc, gaussians.num_gaussians, 0);

        auto cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Copy);

        auto pos_rb_future = cmd_list.ReadBackAsync(gaussians.positions, [&pc_mesh, &gaussians](const void* src_data) {
            const auto* pos_data = reinterpret_cast<const glm::vec3*>(src_data);
            for (uint32_t i = 0; i < gaussians.num_gaussians; ++i)
            {
                pc_mesh.VertexData<glm::vec3>(i, PosAttribIndex) = pos_data[i];
            }
        });
        auto color_rb_future = cmd_list.ReadBackAsync(gaussians.shs, [&pc_mesh, &gaussians](const void* src_data) {
            const auto* sh_data = reinterpret_cast<const glm::vec3*>(src_data);
            const uint32_t num_coeffs = NumCoefficientsFromShDegrees(gaussians.sh_degrees);
            for (uint32_t i = 0; i < gaussians.num_gaussians; ++i)
            {
                pc_mesh.VertexData<glm::vec3>(i, ColorAttribIndex) =
                    glm::clamp(Sh2Rgb(&sh_data[i * num_coeffs]), glm::vec3(0, 0, 0), glm::vec3(1, 1, 1));
            }
        });

        gpu_system.Execute(std::move(cmd_list));

        pos_rb_future.wait();
        color_rb_future.wait();

        SaveMesh(pc_mesh, path);
    }

    Gaussians LoadGaussians(GpuSystem& gpu_system, const std::filesystem::path& path)
    {
        std::ifstream ifs(path, std::ios_base::binary | std::ios_base::in);

        Gaussians ret;

        ifs.read(reinterpret_cast<char*>(&ret.num_gaussians), sizeof(ret.num_gaussians));
        ifs.read(reinterpret_cast<char*>(&ret.sh_degrees), sizeof(ret.sh_degrees));

        const uint32_t num_coeffs = NumCoefficientsFromShDegrees(ret.sh_degrees);

        ret.positions = GpuBuffer(
            gpu_system, ret.num_gaussians * sizeof(glm::vec3), GpuHeap::Default, GpuResourceFlag::UnorderedAccess, "gaussians.positions");
        ret.scales = GpuBuffer(
            gpu_system, ret.num_gaussians * sizeof(glm::vec3), GpuHeap::Default, GpuResourceFlag::UnorderedAccess, "gaussians.scales");
        ret.rotations = GpuBuffer(
            gpu_system, ret.num_gaussians * sizeof(glm::vec4), GpuHeap::Default, GpuResourceFlag::UnorderedAccess, "gaussians.rotations");
        ret.shs = GpuBuffer(gpu_system, ret.num_gaussians * num_coeffs * sizeof(glm::vec3), GpuHeap::Default,
            GpuResourceFlag::UnorderedAccess, "gaussians.shs");
        ret.opacities = GpuBuffer(
            gpu_system, ret.num_gaussians * sizeof(float), GpuHeap::Default, GpuResourceFlag::UnorderedAccess, "gaussians.opacities");

        auto cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Copy);

        cmd_list.Upload(ret.positions,
            [&ifs, &ret](void* dst_data) { ifs.read(reinterpret_cast<char*>(dst_data), ret.num_gaussians * sizeof(glm::vec3)); });
        cmd_list.Upload(ret.scales,
            [&ifs, &ret](void* dst_data) { ifs.read(reinterpret_cast<char*>(dst_data), ret.num_gaussians * sizeof(glm::vec3)); });
        cmd_list.Upload(ret.rotations,
            [&ifs, &ret](void* dst_data) { ifs.read(reinterpret_cast<char*>(dst_data), ret.num_gaussians * sizeof(glm::vec4)); });
        cmd_list.Upload(ret.shs, [&ifs, &ret, num_coeffs](void* dst_data) {
            ifs.read(reinterpret_cast<char*>(dst_data), ret.num_gaussians * num_coeffs * sizeof(glm::vec3));
        });
        cmd_list.Upload(ret.opacities,
            [&ifs, &ret](void* dst_data) { ifs.read(reinterpret_cast<char*>(dst_data), ret.num_gaussians * sizeof(float)); });

        gpu_system.Execute(std::move(cmd_list));

        return ret;
    }

    void SaveGaussians(GpuSystem& gpu_system, const Gaussians& gaussians, const std::filesystem::path& path)
    {
        std::ofstream ofs(path, std::ios_base::binary | std::ios_base::out);

        ofs.write(reinterpret_cast<const char*>(&gaussians.num_gaussians), sizeof(gaussians.num_gaussians));
        ofs.write(reinterpret_cast<const char*>(&gaussians.sh_degrees), sizeof(gaussians.sh_degrees));

        auto cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Copy);

        auto pos_rb_future = cmd_list.ReadBackAsync(gaussians.positions, [&ofs, &gaussians](const void* src_data) {
            ofs.write(reinterpret_cast<const char*>(src_data), gaussians.num_gaussians * sizeof(glm::vec3));
        });
        auto scales_rb_future = cmd_list.ReadBackAsync(gaussians.scales, [&ofs, &gaussians](const void* src_data) {
            ofs.write(reinterpret_cast<const char*>(src_data), gaussians.num_gaussians * sizeof(glm::vec3));
        });
        auto rotations_rb_future = cmd_list.ReadBackAsync(gaussians.rotations, [&ofs, &gaussians](const void* src_data) {
            ofs.write(reinterpret_cast<const char*>(src_data), gaussians.num_gaussians * sizeof(glm::vec4));
        });
        auto shs_rb_future = cmd_list.ReadBackAsync(gaussians.shs, [&ofs, &gaussians](const void* src_data) {
            const uint32_t num_coeffs = NumCoefficientsFromShDegrees(gaussians.sh_degrees);
            ofs.write(reinterpret_cast<const char*>(src_data), gaussians.num_gaussians * num_coeffs * sizeof(glm::vec3));
        });
        auto opacities_rb_future = cmd_list.ReadBackAsync(gaussians.opacities, [&ofs, &gaussians](const void* src_data) {
            ofs.write(reinterpret_cast<const char*>(src_data), gaussians.num_gaussians * sizeof(float));
        });

        gpu_system.Execute(std::move(cmd_list));

        pos_rb_future.wait();
        scales_rb_future.wait();
        rotations_rb_future.wait();
        shs_rb_future.wait();
        opacities_rb_future.wait();
    }
#endif
} // namespace AIHoloImager
