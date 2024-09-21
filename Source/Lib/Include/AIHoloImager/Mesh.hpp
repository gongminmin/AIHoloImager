// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <cstdint>
#include <filesystem>
#include <span>

#include "AIHoloImager/Texture.hpp"

namespace AIHoloImager
{
    struct VertexAttrib
    {
        enum class Semantic
        {
            Position,
            TexCoord,
            Normal,
        };

        static constexpr uint32_t AppendOffset = ~0U;

        Semantic semantic;
        uint32_t index;

        uint32_t channels;

        uint32_t offset = AppendOffset;

        bool operator==(const VertexAttrib& rhs) const;
        bool operator!=(const VertexAttrib& rhs) const
        {
            return !(*this == rhs);
        }
    };

    class VertexDesc
    {
    public:
        static const uint32_t InvalidIndex = ~0U;

    public:
        VertexDesc();
        explicit VertexDesc(std::span<const VertexAttrib> attribs);
        VertexDesc(const VertexDesc& rhs);
        VertexDesc(VertexDesc&& rhs) noexcept;
        ~VertexDesc() noexcept;

        VertexDesc& operator=(const VertexDesc& rhs);
        VertexDesc& operator=(VertexDesc&& rhs) noexcept;

        bool operator==(const VertexDesc& rhs) const;
        bool operator!=(const VertexDesc& rhs) const
        {
            return !(*this == rhs);
        }

        uint32_t Stride() const noexcept;

        std::span<const VertexAttrib> Attribs() const;
        uint32_t FindAttrib(VertexAttrib::Semantic semantic, uint32_t index) const;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    class Mesh
    {
    public:
        Mesh();
        Mesh(VertexDesc vertex_desc, uint32_t num_verts, uint32_t num_indices);
        Mesh(const Mesh& rhs);
        Mesh(Mesh&& rhs) noexcept;
        ~Mesh() noexcept;

        Mesh& operator=(const Mesh& rhs);
        Mesh& operator=(Mesh&& rhs) noexcept;

        bool Valid() const noexcept;

        const VertexDesc& MeshVertexDesc() const noexcept;
        uint32_t NumVertices() const noexcept;
        void ResizeVertices(uint32_t num);

        const float* VertexBuffer() const noexcept;
        void VertexBuffer(const float* data);
        const float* VertexDataPtr(uint32_t index, uint32_t attrib_index) const;
        float* VertexDataPtr(uint32_t index, uint32_t attrib_index);

        template <typename T>
        const T& VertexData(uint32_t index, uint32_t attrib_index) const
        {
            return *reinterpret_cast<const T*>(this->VertexDataPtr(index, attrib_index));
        }
        template <typename T>
        T& VertexData(uint32_t index, uint32_t attrib_index)
        {
            return *reinterpret_cast<T*>(this->VertexDataPtr(index, attrib_index));
        }

        std::span<const uint32_t> Indices() const noexcept;
        void Indices(std::span<const uint32_t> inds);
        void ResizeIndices(uint32_t num);
        uint32_t Index(uint32_t index) const;
        uint32_t& Index(uint32_t index);

        Texture& AlbedoTexture() noexcept;
        const Texture& AlbedoTexture() const noexcept;

        void ResetVertexDesc(VertexDesc new_vertex_desc);

        void ComputeNormals();

        Mesh ExtractMesh(VertexDesc new_vertex_desc, std::span<const uint32_t> extract_indices) const;
        Mesh ExtractMesh(
            VertexDesc new_vertex_desc, std::span<const uint32_t> new_vertex_reference, std::span<const uint32_t> new_indices) const;

        Mesh UnwrapUv(uint32_t texture_size, uint32_t padding, std::vector<uint32_t>* vertex_referencing = nullptr);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    Mesh LoadMesh(const std::filesystem::path& path);
    void SaveMesh(const Mesh& mesh, const std::filesystem::path& path);
} // namespace AIHoloImager
