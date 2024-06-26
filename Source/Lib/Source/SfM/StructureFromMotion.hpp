// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4127) // Ignore conditional expression is constant
    #pragma warning(disable : 5054) // Ignore operator between enums of different types
#endif
#include <Eigen/Core>
#ifdef _MSC_VER
    #pragma warning(pop)
#endif

#include "AIHoloImager/Texture.hpp"
#include "Util/Noncopyable.hpp"

namespace AIHoloImager
{
    class StructureFromMotion
    {
        DISALLOW_COPY_AND_ASSIGN(StructureFromMotion);

    public:
        struct View
        {
            Texture image;

            uint32_t intrinsic_id;

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> rotation;
            Eigen::Vector3d center;
        };

        struct PinholeIntrinsic
        {
            uint32_t width;
            uint32_t height;

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> k;
        };

        struct Observation
        {
            uint32_t view_id;
            Eigen::Vector2d point;
            uint32_t feat_id;
        };

        struct Landmark
        {
            Eigen::Vector3d point;
            std::vector<Observation> obs;
        };

        struct Result
        {
            std::vector<View> views;
            std::vector<PinholeIntrinsic> intrinsics;

            std::vector<Landmark> structure;
        };

    public:
        explicit StructureFromMotion(const std::filesystem::path& exe_dir);
        StructureFromMotion(StructureFromMotion&& other) noexcept;
        ~StructureFromMotion() noexcept;

        StructureFromMotion& operator=(StructureFromMotion&& other) noexcept;

        Result Process(const std::filesystem::path& input_path, bool sequential, const std::filesystem::path& tmp_dir);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
