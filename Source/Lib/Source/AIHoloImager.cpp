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

#include "MeshRecon//MeshReconstruction.hpp"
#include "SfM/StructureFromMotion.hpp"

namespace
{
    std::filesystem::path ExePath()
    {
        char exe_path[MAX_PATH];
        GetModuleFileNameA(nullptr, exe_path, sizeof(exe_path));
        return std::filesystem::path(exe_path);
    }
} // namespace

namespace AIHoloImager
{
    class AIHoloImager::Impl
    {
    public:
        Impl(const std::filesystem::path& tmp_dir) : exe_dir_(ExePath()), tmp_dir_(tmp_dir), sfm_(exe_dir_), mesh_recon_(exe_dir_)
        {
        }

        Mesh Generate(const std::filesystem::path& input_path)
        {
            const auto sfm_result = sfm_.Process(input_path, true, tmp_dir_);
            mesh_recon_.Process(sfm_result, tmp_dir_);
            return Mesh();
        }

    private:
        std::filesystem::path exe_dir_;
        std::filesystem::path tmp_dir_;

        StructureFromMotion sfm_;
        MeshReconstruction mesh_recon_;
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
