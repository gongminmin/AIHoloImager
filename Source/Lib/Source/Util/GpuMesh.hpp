// Copyright (c) 2026 Minmin Gong
//

#pragma once

#include "AIHoloImager/Mesh.hpp"
#include "Base/Noncopyable.hpp"
#include "Gpu/GpuBuffer.hpp"
#include "Gpu/GpuTexture.hpp"
#include "Gpu/GpuVertexLayout.hpp"

namespace AIHoloImager
{
    class GpuMesh
    {
        DISALLOW_COPY_AND_ASSIGN(GpuMesh)

    public:
        GpuMesh() noexcept;
        GpuMesh(GpuVertexLayout vertex_desc, GpuFormat index_format);
        ~GpuMesh() noexcept;

        GpuMesh(GpuMesh&& rhs) noexcept;
        GpuMesh& operator=(GpuMesh&& rhs) noexcept;

        bool Valid() const noexcept;

        const GpuVertexLayout& MeshVertexDesc() const noexcept;
        GpuFormat IndexFormat() const noexcept;

        GpuBuffer& VertexBuffer() noexcept;
        const GpuBuffer& VertexBuffer() const noexcept;
        GpuBuffer& IndexBuffer() noexcept;
        const GpuBuffer& IndexBuffer() const noexcept;

        GpuTexture2D& AlbedoTexture() noexcept;
        const GpuTexture2D& AlbedoTexture() const noexcept;

        uint32_t NumVertices() const noexcept;
        uint32_t NumIndices() const noexcept;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    Mesh ToMesh(GpuSystem& gpu_system, const GpuMesh& gpu_mesh);
    GpuMesh ToGpuMesh(GpuSystem& gpu_system, const Mesh& mesh);
} // namespace AIHoloImager
