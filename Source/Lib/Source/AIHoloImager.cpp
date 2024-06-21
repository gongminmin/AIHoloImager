// Copyright (c) 2024 Minmin Gong
//

#include "AIHoloImager/AIHoloImager.hpp"

#include "SfM/StructureFromMotion.hpp"

namespace AIHoloImager
{
    class AIHoloImager::Impl
    {
    public:
        Impl(const std::filesystem::path& tmp_dir) : tmp_dir_(tmp_dir)
        {
        }

        Mesh Generate(const std::filesystem::path& input_path)
        {
            sfm_.Process(input_path, true, tmp_dir_);
            return Mesh();
        }

    private:
        std::filesystem::path tmp_dir_;

        StructureFromMotion sfm_;
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
