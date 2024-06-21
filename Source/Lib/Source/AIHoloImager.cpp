// Copyright (c) 2024 Minmin Gong
//

#include "AIHoloImager/AIHoloImager.hpp"

#include "SfM/StructureFromMotion.hpp"

namespace AIHoloImager
{
    class AIHoloImager::Impl
    {
    public:
        Mesh Generate(const std::filesystem::path& input_path)
        {
            sfm_.Process(input_path, true);
            return Mesh();
        }

    private:
        StructureFromMotion sfm_;
    };

    AIHoloImager::AIHoloImager() : impl_(std::make_unique<Impl>())
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
