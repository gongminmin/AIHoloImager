// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <cstdint>
#include <filesystem>
#include <span>

#include <DirectXMath.h>

#include "AIHoloImager/Common.hpp"
#include "AIHoloImager/Texture.hpp"

namespace AIHoloImager
{
    class Mesh
    {
    public:
        struct VertexFormat
        {
            DirectX::XMFLOAT3 pos;
            DirectX::XMFLOAT2 texcoord;
        };

    public:
        AIHI_API Mesh();
        AIHI_API Mesh(uint32_t num_verts, uint32_t num_indices);
        AIHI_API Mesh(const Mesh& rhs);
        AIHI_API Mesh(Mesh&& rhs) noexcept;
        AIHI_API ~Mesh() noexcept;

        AIHI_API Mesh& operator=(const Mesh& rhs);
        AIHI_API Mesh& operator=(Mesh&& rhs) noexcept;

        AIHI_API bool Valid() const noexcept;

        AIHI_API std::span<const VertexFormat> Vertices() const noexcept;
        AIHI_API void Vertices(std::span<const VertexFormat> verts);
        AIHI_API void ResizeVertices(uint32_t num);
        AIHI_API const VertexFormat& Vertex(uint32_t index) const;
        AIHI_API VertexFormat& Vertex(uint32_t index);

        AIHI_API std::span<const uint32_t> Indices() const noexcept;
        AIHI_API void Indices(std::span<const uint32_t> inds);
        AIHI_API void ResizeIndices(uint32_t num);
        AIHI_API uint32_t Index(uint32_t index) const;
        AIHI_API uint32_t& Index(uint32_t index);

        AIHI_API const Texture& AlbedoTexture() const noexcept;
        AIHI_API void AlbedoTexture(Texture tex);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    AIHI_API Mesh LoadMesh(const std::filesystem::path& path);
    AIHI_API void SaveMesh(const Mesh& mesh, const std::filesystem::path& path);
} // namespace AIHoloImager
