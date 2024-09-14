// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <cstdint>
#include <filesystem>
#include <span>

#include <DirectXMath.h>

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
        Mesh();
        Mesh(uint32_t num_verts, uint32_t num_indices);
        Mesh(const Mesh& rhs);
        Mesh(Mesh&& rhs) noexcept;
        ~Mesh() noexcept;

        Mesh& operator=(const Mesh& rhs);
        Mesh& operator=(Mesh&& rhs) noexcept;

        bool Valid() const noexcept;

        std::span<const VertexFormat> Vertices() const noexcept;
        void Vertices(std::span<const VertexFormat> verts);
        void ResizeVertices(uint32_t num);
        const VertexFormat& Vertex(uint32_t index) const;
        VertexFormat& Vertex(uint32_t index);

        std::span<const uint32_t> Indices() const noexcept;
        void Indices(std::span<const uint32_t> inds);
        void ResizeIndices(uint32_t num);
        uint32_t Index(uint32_t index) const;
        uint32_t& Index(uint32_t index);

        Texture& AlbedoTexture() noexcept;
        const Texture& AlbedoTexture() const noexcept;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    Mesh LoadMesh(const std::filesystem::path& path);
    void SaveMesh(const Mesh& mesh, const std::filesystem::path& path);

    Mesh UnwrapUv(const Mesh& mesh, uint32_t texture_size, uint32_t padding, std::vector<uint32_t>& vertex_referencing);
} // namespace AIHoloImager
