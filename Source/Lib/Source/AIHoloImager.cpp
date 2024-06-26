// Copyright (c) 2024 Minmin Gong
//

#include "AIHoloImager/AIHoloImager.hpp"

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#endif

#include "Gpu/GpuSystem.hpp"
#include "MeshRecon//MeshReconstruction.hpp"
#include "MvRenderer/MultiViewRenderer.hpp"
#include "SfM/StructureFromMotion.hpp"

namespace
{
    std::filesystem::path ExeDir()
    {
        char exe_path[MAX_PATH];
        GetModuleFileNameA(nullptr, exe_path, sizeof(exe_path));
        return std::filesystem::path(exe_path).parent_path();
    }
} // namespace

namespace AIHoloImager
{
    class AIHoloImager::Impl
    {
    public:
        explicit Impl(const std::filesystem::path& tmp_dir)
            : exe_dir_(ExeDir()), tmp_dir_(tmp_dir), sfm_(exe_dir_), mesh_recon_(exe_dir_), mv_renderer_(gpu_system_, 320, 320)
        {
        }

        Mesh Generate(const std::filesystem::path& input_path)
        {
            const auto sfm_result = sfm_.Process(input_path, true, tmp_dir_);
            const auto mesh_recon_result = mesh_recon_.Process(sfm_result, true, 2048, tmp_dir_);
            const auto mv_renderer_result = mv_renderer_.Render(mesh_recon_result.mesh);
            return Mesh();
        }

    private:
        std::filesystem::path exe_dir_;
        std::filesystem::path tmp_dir_;

        GpuSystem gpu_system_;
        StructureFromMotion sfm_;
        MeshReconstruction mesh_recon_;
        MultiViewRenderer mv_renderer_;
    };

    AIHoloImager::AIHoloImager(const std::filesystem::path& tmp_dir) : impl_(std::make_unique<Impl>(tmp_dir))
    {
    }
    AIHoloImager::AIHoloImager(AIHoloImager&& rhs) noexcept = default;
    AIHoloImager::~AIHoloImager() noexcept = default;

    AIHoloImager& AIHoloImager::operator=(AIHoloImager&& rhs) noexcept = default;

    Mesh AIHoloImager::Generate(const std::filesystem::path& input_path)
    {
        return impl_->Generate(input_path);
    }
} // namespace AIHoloImager
