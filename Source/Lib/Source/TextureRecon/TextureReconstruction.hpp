// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <filesystem>

#include <DirectXCollision.h>
#include <DirectXMath.h>

#include "AIHoloImager/Mesh.hpp"
#include "Gpu/GpuSystem.hpp"
#include "Gpu/GpuTexture.hpp"
#include "SfM/StructureFromMotion.hpp"
#include "Util/Noncopyable.hpp"

namespace AIHoloImager
{
    class TextureReconstruction
    {
        DISALLOW_COPY_AND_ASSIGN(TextureReconstruction);

    public:
        struct Result
        {
            GpuTexture2D color_tex;

            GpuBuffer counter_buff;
            GpuBuffer uv_buff;
            GpuBuffer pos_buff;
        };

    public:
        TextureReconstruction(const std::filesystem::path& exe_dir, GpuSystem& gpu_system);
        TextureReconstruction(TextureReconstruction&& other) noexcept;
        ~TextureReconstruction() noexcept;

        TextureReconstruction& operator=(TextureReconstruction&& other) noexcept;

        Result Process(const Mesh& pos_only_mesh, const Mesh& pos_uv_mesh, const std::vector<uint32_t>& vertex_referencing,
            const DirectX::XMMATRIX& model_mtx, const DirectX::BoundingOrientedBox& world_obb, const StructureFromMotion::Result& sfm_input,
            uint32_t texture_size, bool empty_pos, const std::filesystem::path& tmp_dir);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
