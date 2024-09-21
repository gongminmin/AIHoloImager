// Copyright (c) 2024 Minmin Gong
//

#include "AIHoloImager/Mesh.hpp"

#include <set>

#include <assimp/Exporter.hpp>
#include <assimp/GltfMaterial.h>
#include <assimp/Importer.hpp>
#include <assimp/material.h>
#include <assimp/scene.h>
#include <xatlas/xatlas.h>

using namespace DirectX;

namespace AIHoloImager
{
    bool VertexAttrib::operator==(const VertexAttrib& rhs) const
    {
        return std::memcmp(this, &rhs, sizeof(rhs)) == 0;
    }

    class VertexDesc::Impl
    {
    public:
        explicit Impl(std::span<const VertexAttrib> attribs) : attribs_(attribs.begin(), attribs.end())
        {
            for (size_t i = 0; i < attribs_.size(); ++i)
            {
                if (attribs_[i].offset == VertexAttrib::AppendOffset)
                {
                    if (i == 0)
                    {
                        attribs_[i].offset = 0;
                    }
                    else
                    {
                        attribs_[i].offset = attribs_[i - 1].offset + attribs_[i - 1].channels * sizeof(float);
                    }
                }
            }

            stride_ = 0;
            for (size_t i = 0; i < attribs_.size(); ++i)
            {
                stride_ = std::max(stride_, static_cast<uint32_t>(attribs_[i].offset + attribs_[i].channels * sizeof(float)));
            }
        }

        uint32_t Stride() const noexcept
        {
            return stride_;
        }

        std::span<const VertexAttrib> Attribs() const
        {
            return std::span(attribs_);
        }

        uint32_t FindAttrib(VertexAttrib::Semantic semantic, uint32_t index) const
        {
            for (size_t i = 0; i < attribs_.size(); ++i)
            {
                if ((attribs_[i].semantic == semantic) && (attribs_[i].index == index))
                {
                    return static_cast<uint32_t>(i);
                }
            }

            return InvalidIndex;
        }

        bool operator==(const Impl& rhs) const
        {
            return attribs_ == rhs.attribs_;
        }

    private:
        std::vector<VertexAttrib> attribs_;
        uint32_t stride_;
    };

    VertexDesc::VertexDesc() = default;
    VertexDesc::VertexDesc(std::span<const VertexAttrib> attribs) : impl_(std::make_unique<Impl>(std::move(attribs)))
    {
    }
    VertexDesc::VertexDesc(const VertexDesc& rhs) : impl_(rhs.impl_ ? std::make_unique<Impl>(*rhs.impl_) : nullptr)
    {
    }
    VertexDesc::VertexDesc(VertexDesc&& rhs) noexcept = default;
    VertexDesc::~VertexDesc() noexcept = default;

    VertexDesc& VertexDesc::operator=(const VertexDesc& rhs)
    {
        if (this != &rhs)
        {
            if (rhs.impl_)
            {
                if (impl_)
                {
                    *impl_ = *rhs.impl_;
                }
                else
                {
                    impl_ = std::make_unique<Impl>(*rhs.impl_);
                }
            }
            else
            {
                impl_.reset();
            }
        }
        return *this;
    }
    VertexDesc& VertexDesc::operator=(VertexDesc&& rhs) noexcept = default;

    uint32_t VertexDesc::Stride() const noexcept
    {
        return impl_->Stride();
    }

    bool VertexDesc::operator==(const VertexDesc& rhs) const
    {
        return *impl_ == *rhs.impl_;
    }

    std::span<const VertexAttrib> VertexDesc::Attribs() const
    {
        return impl_->Attribs();
    }

    uint32_t VertexDesc::FindAttrib(VertexAttrib::Semantic semantic, uint32_t index) const
    {
        return impl_->FindAttrib(semantic, index);
    }

    class Mesh::Impl
    {
    public:
        Impl(VertexDesc vertex_desc, uint32_t num_verts, uint32_t num_indices) : vertex_desc_(std::move(vertex_desc)), indices_(num_indices)
        {
            this->ResizeVertices(num_verts);
        }

        const VertexDesc& MeshVertexDesc() const noexcept
        {
            return vertex_desc_;
        }

        uint32_t NumVertices() const noexcept
        {
            return num_vertices_;
        }
        void ResizeVertices(uint32_t num)
        {
            num_vertices_ = num;
            vertices_.resize(num * vertex_desc_.Stride());
        }

        const float* VertexBuffer() const noexcept
        {
            return vertices_.data();
        }
        void VertexBuffer(const float* data)
        {
            std::memcpy(vertices_.data(), data, num_vertices_ * vertex_desc_.Stride());
        }
        const float* VertexDataPtr(uint32_t index, uint32_t attrib_index) const
        {
            return const_cast<Impl*>(this)->VertexDataPtr(index, attrib_index);
        }
        float* VertexDataPtr(uint32_t index, uint32_t attrib_index)
        {
            const auto& vertex_attrib = vertex_desc_.Attribs()[attrib_index];
            const uint32_t offset = vertex_attrib.offset / sizeof(float);
            return &vertices_[index * (vertex_desc_.Stride() / sizeof(float)) + offset];
        }

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

        std::span<const uint32_t> Indices() const noexcept
        {
            return std::span<const uint32_t>(indices_.begin(), indices_.end());
        }
        void Indices(std::span<const uint32_t> inds)
        {
            indices_.assign(inds.begin(), inds.end());
        }
        void ResizeIndices(uint32_t num)
        {
            indices_.resize(num);
        }
        uint32_t Index(uint32_t index) const
        {
            return indices_[index];
        }
        uint32_t& Index(uint32_t index)
        {
            return indices_[index];
        }

        Texture& AlbedoTexture() noexcept
        {
            return albedo_tex_;
        }
        const Texture& AlbedoTexture() const noexcept
        {
            return albedo_tex_;
        }

        void ResetVertexDesc(VertexDesc new_vertex_desc)
        {
            if (vertex_desc_ == new_vertex_desc)
            {
                return;
            }

            const auto old_vertex_attribs = vertex_desc_.Attribs();
            const auto new_vertex_attribs = new_vertex_desc.Attribs();
            const uint32_t old_stride = vertex_desc_.Stride();
            const uint32_t new_stride = new_vertex_desc.Stride();

            std::vector<float> new_vertices(num_vertices_ * new_stride / sizeof(float));

            const std::byte* old_vertex_data = reinterpret_cast<const std::byte*>(this->VertexDataPtr(0, 0));
            std::byte* new_vertex_data = reinterpret_cast<std::byte*>(new_vertices.data());
            if ((new_vertex_attribs.size() > old_vertex_attribs.size()) &&
                std::equal(new_vertex_attribs.begin(), new_vertex_attribs.begin() + old_vertex_attribs.size(), old_vertex_attribs.begin()))
            {
                for (uint32_t vi = 0; vi < num_vertices_; ++vi)
                {
                    std::memcpy(new_vertex_data, old_vertex_data, old_stride);
                    std::memset(new_vertex_data + old_stride, 0, new_stride - old_stride);

                    old_vertex_data += old_stride;
                    new_vertex_data += new_stride;
                }
            }
            else
            {
                for (uint32_t vi = 0; vi < num_vertices_; ++vi)
                {
                    for (const auto& new_va : new_vertex_attribs)
                    {
                        bool found = false;
                        for (const auto& old_va : old_vertex_attribs)
                        {
                            if ((new_va.semantic == old_va.semantic) && (new_va.index == old_va.index))
                            {
                                std::memcpy(new_vertex_data + new_va.offset, old_vertex_data + old_va.offset,
                                    std::min(new_va.channels, old_va.channels) * sizeof(float));
                                found = true;
                                break;
                            }
                        }

                        if (!found)
                        {
                            std::memset(new_vertex_data + new_va.offset, 0, new_va.channels * sizeof(float));
                        }
                    }

                    old_vertex_data += old_stride;
                    new_vertex_data += new_stride;
                }
            }

            vertices_ = std::move(new_vertices);
            vertex_desc_ = std::move(new_vertex_desc);
        }

        void ComputeNormals()
        {
            uint32_t normal_attrib_index = vertex_desc_.FindAttrib(VertexAttrib::Semantic::Normal, 0);
            if (normal_attrib_index == VertexDesc::InvalidIndex)
            {
                const auto old_vertex_attrib = vertex_desc_.Attribs();
                std::vector<VertexAttrib> new_vertex_attribs(old_vertex_attrib.begin(), old_vertex_attrib.end());
                normal_attrib_index = static_cast<uint32_t>(new_vertex_attribs.size());
                new_vertex_attribs.push_back({VertexAttrib::Semantic::Normal, 0, 3});

                this->ResetVertexDesc(VertexDesc(new_vertex_attribs));
            }

            for (uint32_t i = 0; i < static_cast<uint32_t>(indices_.size()); i += 3)
            {
                const uint32_t ind[] = {indices_[i + 0], indices_[i + 1], indices_[i + 2]};
                const XMVECTOR pos[] = {
                    XMLoadFloat3(&this->VertexData<XMFLOAT3>(ind[0], 0)),
                    XMLoadFloat3(&this->VertexData<XMFLOAT3>(ind[1], 0)),
                    XMLoadFloat3(&this->VertexData<XMFLOAT3>(ind[2], 0)),
                };
                const XMVECTOR edge[] = {pos[1] - pos[0], pos[2] - pos[0]};
                const XMVECTOR face_normal = XMVector3Cross(edge[0], edge[1]);
                for (uint32_t j = 0; j < 3; ++j)
                {
                    XMFLOAT3& normal3 = this->VertexData<XMFLOAT3>(ind[j], normal_attrib_index);
                    XMVECTOR normal = XMLoadFloat3(&normal3);
                    normal += face_normal;
                    XMStoreFloat3(&normal3, normal);
                }
            }

            for (uint32_t vi = 0; vi < num_vertices_; ++vi)
            {
                XMFLOAT3& normal3 = this->VertexData<XMFLOAT3>(vi, normal_attrib_index);
                XMVECTOR normal = XMLoadFloat3(&normal3);
                normal = XMVector3Normalize(normal);
                XMStoreFloat3(&normal3, normal);
            }
        }

        Mesh ExtractMesh(VertexDesc new_vertex_desc, std::span<const uint32_t> extract_indices) const
        {
            std::set<uint32_t> unique_indices(extract_indices.begin(), extract_indices.end());

            std::vector<uint32_t> vert_mapping(num_vertices_, ~0U);
            std::vector<uint32_t> new_vertex_references(unique_indices.size());
            uint32_t new_index = 0;
            for (const uint32_t vi : unique_indices)
            {
                vert_mapping[vi] = new_index;
                new_vertex_references[new_index] = vi;
                ++new_index;
            }

            std::vector<uint32_t> new_indices(extract_indices.size());
            for (size_t i = 0; i < extract_indices.size(); ++i)
            {
                new_indices[i] = vert_mapping[extract_indices[i]];
                assert(new_indices[i] != ~0U);
            }

            return this->ExtractMesh(std::move(new_vertex_desc), new_vertex_references, new_indices);
        }

        Mesh ExtractMesh(
            VertexDesc new_vertex_desc, std::span<const uint32_t> new_vertex_references, std::span<const uint32_t> new_indices) const
        {
            const uint32_t new_num_vertices = static_cast<uint32_t>(new_vertex_references.size());
            Mesh ret(std::move(new_vertex_desc), new_num_vertices, static_cast<uint32_t>(new_indices.size()));

            if (vertex_desc_ == ret.MeshVertexDesc())
            {
                const uint32_t vertex_size = ret.MeshVertexDesc().Stride();
                for (uint32_t vi = 0; vi < new_num_vertices; ++vi)
                {
                    std::memcpy(ret.VertexDataPtr(vi, 0), this->VertexDataPtr(new_vertex_references[vi], 0), vertex_size);
                }
            }
            else
            {
                const auto old_vertex_attribs = vertex_desc_.Attribs();
                const auto new_vertex_attribs = ret.MeshVertexDesc().Attribs();
                for (uint32_t ai = 0; ai < new_vertex_attribs.size(); ++ai)
                {
                    const uint32_t old_ai = vertex_desc_.FindAttrib(new_vertex_attribs[ai].semantic, new_vertex_attribs[ai].index);
                    if (old_ai != VertexDesc::InvalidIndex)
                    {
                        const uint32_t attrib_size =
                            std::min(new_vertex_attribs[ai].channels, old_vertex_attribs[old_ai].channels) * sizeof(float);
                        for (uint32_t vi = 0; vi < new_num_vertices; ++vi)
                        {
                            std::memcpy(ret.VertexDataPtr(vi, ai), this->VertexDataPtr(new_vertex_references[vi], old_ai), attrib_size);
                        }
                    }
                    else
                    {
                        const uint32_t attrib_size = new_vertex_attribs[ai].channels * sizeof(float);
                        for (uint32_t vi = 0; vi < new_num_vertices; ++vi)
                        {
                            std::memset(ret.VertexDataPtr(vi, ai), 0, attrib_size);
                        }
                    }
                }
            }

            ret.impl_->indices_.assign(new_indices.begin(), new_indices.end());
            ret.impl_->albedo_tex_ = albedo_tex_;
            return ret;
        }

    private:
        VertexDesc vertex_desc_;
        std::vector<float> vertices_;
        uint32_t num_vertices_ = 0;

        std::vector<uint32_t> indices_;

        Texture albedo_tex_;
    };

    Mesh::Mesh() = default;
    Mesh::Mesh(VertexDesc vertex_desc, uint32_t num_verts, uint32_t num_indices)
        : impl_(std::make_unique<Impl>(std::move(vertex_desc), num_verts, num_indices))
    {
    }
    Mesh::Mesh(const Mesh& rhs) : impl_(rhs.impl_ ? std::make_unique<Impl>(*rhs.impl_) : nullptr)
    {
    }
    Mesh::Mesh(Mesh&& rhs) noexcept = default;
    Mesh::~Mesh() noexcept = default;

    Mesh& Mesh::operator=(const Mesh& rhs)
    {
        if (this != &rhs)
        {
            if (rhs.impl_)
            {
                if (impl_)
                {
                    *impl_ = *rhs.impl_;
                }
                else
                {
                    impl_ = std::make_unique<Impl>(*rhs.impl_);
                }
            }
            else
            {
                impl_.reset();
            }
        }
        return *this;
    }
    Mesh& Mesh::operator=(Mesh&& rhs) noexcept = default;

    bool Mesh::Valid() const noexcept
    {
        return static_cast<bool>(impl_);
    }

    const VertexDesc& Mesh::MeshVertexDesc() const noexcept
    {
        return impl_->MeshVertexDesc();
    }
    uint32_t Mesh::NumVertices() const noexcept
    {
        return impl_->NumVertices();
    }
    void Mesh::ResizeVertices(uint32_t num)
    {
        impl_->ResizeVertices(num);
    }

    const float* Mesh::VertexBuffer() const noexcept
    {
        return impl_->VertexBuffer();
    }
    void Mesh::VertexBuffer(const float* data)
    {
        impl_->VertexBuffer(data);
    }
    const float* Mesh::VertexDataPtr(uint32_t index, uint32_t attrib_index) const
    {
        return impl_->VertexDataPtr(index, attrib_index);
    }
    float* Mesh::VertexDataPtr(uint32_t index, uint32_t attrib_index)
    {
        return impl_->VertexDataPtr(index, attrib_index);
    }

    std::span<const uint32_t> Mesh::Indices() const noexcept
    {
        return impl_->Indices();
    }
    void Mesh::Indices(std::span<const uint32_t> inds)
    {
        impl_->Indices(std::move(inds));
    }
    void Mesh::ResizeIndices(uint32_t num)
    {
        impl_->ResizeIndices(num);
    }
    uint32_t Mesh::Index(uint32_t index) const
    {
        return impl_->Index(index);
    }
    uint32_t& Mesh::Index(uint32_t index)
    {
        return impl_->Index(index);
    }

    Texture& Mesh::AlbedoTexture() noexcept
    {
        return impl_->AlbedoTexture();
    }
    const Texture& Mesh::AlbedoTexture() const noexcept
    {
        return impl_->AlbedoTexture();
    }

    void Mesh::ResetVertexDesc(VertexDesc new_vertex_desc)
    {
        impl_->ResetVertexDesc(std::move(new_vertex_desc));
    }

    void Mesh::ComputeNormals()
    {
        return impl_->ComputeNormals();
    }

    Mesh Mesh::ExtractMesh(VertexDesc new_vertex_desc, std::span<const uint32_t> extract_indices) const
    {
        return impl_->ExtractMesh(std::move(new_vertex_desc), std::move(extract_indices));
    }

    Mesh Mesh::ExtractMesh(
        VertexDesc new_vertex_desc, std::span<const uint32_t> new_vertex_references, std::span<const uint32_t> new_indices) const
    {
        return impl_->ExtractMesh(std::move(new_vertex_desc), std::move(new_vertex_references), std::move(new_indices));
    }

    Mesh LoadMesh(const std::filesystem::path& path)
    {
        Assimp::Importer importer;
        importer.SetPropertyInteger(AI_CONFIG_IMPORT_TER_MAKE_UVS, 1);
        importer.SetPropertyFloat(AI_CONFIG_PP_GSN_MAX_SMOOTHING_ANGLE, 80);
        importer.SetPropertyInteger(AI_CONFIG_PP_SBP_REMOVE, 0);
        importer.SetPropertyInteger(AI_CONFIG_GLOB_MEASURE_TIME, 1);

        const aiScene* ai_scene = importer.ReadFile(path.string().c_str(), 0);

        Mesh mesh;
        if (ai_scene)
        {
            const aiMesh* ai_mesh = ai_scene->mMeshes[0];

            std::vector<VertexAttrib> vertex_attribs;
            vertex_attribs.push_back({VertexAttrib::Semantic::Position, 0, 3});
            if (ai_mesh->HasNormals())
            {
                vertex_attribs.push_back({VertexAttrib::Semantic::Normal, 0, 3});
            }
            for (uint32_t i = 0; i < ai_mesh->GetNumUVChannels(); ++i)
            {
                vertex_attribs.push_back({VertexAttrib::Semantic::TexCoord, i, ai_mesh->mNumUVComponents[i]});
            }

            mesh = Mesh(VertexDesc(vertex_attribs), ai_mesh->mNumVertices, ai_mesh->mNumFaces * 3);

            const aiMaterial* mtl = ai_scene->mMaterials[ai_mesh->mMaterialIndex];
            unsigned int count = aiGetMaterialTextureCount(mtl, aiTextureType_DIFFUSE);
            if (count > 0)
            {
                aiString str;
                aiGetMaterialTexture(mtl, aiTextureType_DIFFUSE, 0, &str, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
                mesh.AlbedoTexture() = LoadTexture(path.parent_path() / str.C_Str());
            }

            const auto& vertex_desc = mesh.MeshVertexDesc();
            const uint32_t pos_attrib_index = vertex_desc.FindAttrib(VertexAttrib::Semantic::Position, 0);
            const uint32_t normal_attrib_index = vertex_desc.FindAttrib(VertexAttrib::Semantic::Normal, 0);
            std::vector<uint32_t> texcoord_attrib_indices(ai_mesh->GetNumUVChannels());
            for (uint32_t i = 0; i < ai_mesh->GetNumUVChannels(); ++i)
            {
                texcoord_attrib_indices[i] = vertex_desc.FindAttrib(VertexAttrib::Semantic::TexCoord, i);
            }
            for (unsigned int vi = 0; vi < ai_mesh->mNumVertices; ++vi)
            {
                if (pos_attrib_index != VertexDesc::InvalidIndex)
                {
                    mesh.VertexData<XMFLOAT3>(vi, pos_attrib_index) = XMFLOAT3(&ai_mesh->mVertices[vi].x);
                }
                if (normal_attrib_index != VertexDesc::InvalidIndex)
                {
                    mesh.VertexData<XMFLOAT3>(vi, normal_attrib_index) = XMFLOAT3(&ai_mesh->mNormals[vi].x);
                }
                for (size_t j = 0; j < texcoord_attrib_indices.size(); ++j)
                {
                    std::memcpy(mesh.VertexDataPtr(vi, texcoord_attrib_indices[j]), &ai_mesh->mTextureCoords[j][vi].x,
                        ai_mesh->mNumUVComponents[j] * sizeof(float));
                }
            }

            for (unsigned int fi = 0; fi < ai_mesh->mNumFaces; ++fi)
            {
                for (uint32_t vi = 0; vi < 3; ++vi)
                {
                    mesh.Index(fi * 3 + vi) = ai_mesh->mFaces[fi].mIndices[vi];
                }
            }
        }

        return mesh;
    }

    void SaveMesh(const Mesh& mesh, const std::filesystem::path& path)
    {
        if (!mesh.Valid())
        {
            return;
        }

        const auto ext = path.extension();
        const bool is_gltf = (ext == ".gltf") || (ext == ".glb");

        aiScene ai_scene;

        ai_scene.mNumMaterials = 1;
        ai_scene.mMaterials = new aiMaterial*[ai_scene.mNumMaterials];
        ai_scene.mMaterials[0] = new aiMaterial;
        auto& ai_mtl = *ai_scene.mMaterials[0];

        {
            aiString name;
            name.Set("Diffuse");
            ai_mtl.AddProperty(&name, AI_MATKEY_NAME);

            if (is_gltf)
            {
                aiColor4D const ai_albedo(1, 1, 1, 1);
                ai_mtl.AddProperty(&ai_albedo, 1, AI_MATKEY_BASE_COLOR);
            }
            else
            {
                const aiColor3D ai_diffuse(1, 1, 1);
                ai_mtl.AddProperty(&ai_diffuse, 1, AI_MATKEY_COLOR_DIFFUSE);
            }

            const ai_real ai_opacity = 1;
            ai_mtl.AddProperty(&ai_opacity, 1, AI_MATKEY_OPACITY);
        }

        if (is_gltf)
        {
            aiString ai_alpha_mode("OPAQUE");
            ai_mtl.AddProperty(&ai_alpha_mode, AI_MATKEY_GLTF_ALPHAMODE);
        }

        if (mesh.AlbedoTexture().Valid())
        {
            const std::filesystem::path albedo_tex_path = path.stem().string() + "_0.png";
            SaveTexture(mesh.AlbedoTexture(), path.parent_path() / albedo_tex_path);

            {
                aiString name;
                name.Set(albedo_tex_path.string());
                if (is_gltf)
                {
                    ai_mtl.AddProperty(&name, _AI_MATKEY_TEXTURE_BASE, aiTextureType_BASE_COLOR);
                }
                else
                {
                    ai_mtl.AddProperty(&name, AI_MATKEY_TEXTURE_DIFFUSE(0));
                }
            }
        }

        ai_scene.mNumMeshes = 1;
        ai_scene.mMeshes = new aiMesh*[ai_scene.mNumMeshes];
        ai_scene.mMeshes[0] = new aiMesh;
        auto& ai_mesh = *ai_scene.mMeshes[0];

        ai_mesh.mMaterialIndex = 0;
        ai_mesh.mPrimitiveTypes = aiPrimitiveType_TRIANGLE;

        ai_mesh.mName.Set(path.stem().string().c_str());

        ai_mesh.mNumVertices = static_cast<uint32_t>(mesh.NumVertices());

        const auto& vertex_desc = mesh.MeshVertexDesc();
        const uint32_t pos_attrib = vertex_desc.FindAttrib(VertexAttrib::Semantic::Position, 0);
        const uint32_t normal_attrib = vertex_desc.FindAttrib(VertexAttrib::Semantic::Normal, 0);
        std::vector<uint32_t> texcoord_attribs;
        for (uint32_t i = 0; i < 8; ++i)
        {
            const uint32_t texcoord_attrib = vertex_desc.FindAttrib(VertexAttrib::Semantic::TexCoord, i);
            if (texcoord_attrib == VertexDesc::InvalidIndex)
            {
                break;
            }

            texcoord_attribs.push_back(texcoord_attrib);
        }

        if (pos_attrib != VertexDesc::InvalidIndex)
        {
            ai_mesh.mVertices = new aiVector3D[ai_mesh.mNumVertices];
        }
        if (normal_attrib != VertexDesc::InvalidIndex)
        {
            ai_mesh.mNormals = new aiVector3D[ai_mesh.mNumVertices];
        }
        for (uint32_t i = 0; i < texcoord_attribs.size(); ++i)
        {
            if (texcoord_attribs[i] != VertexDesc::InvalidIndex)
            {
                ai_mesh.mTextureCoords[i] = new aiVector3D[ai_mesh.mNumVertices];
                ai_mesh.mNumUVComponents[i] = vertex_desc.Attribs()[texcoord_attribs[i]].channels;
            }
        }

        for (uint32_t i = 0; i < ai_mesh.mNumVertices; ++i)
        {
            if (pos_attrib != VertexDesc::InvalidIndex)
            {
                ai_mesh.mVertices[i] = mesh.VertexData<aiVector3D>(i, pos_attrib);
            }
            if (normal_attrib != VertexDesc::InvalidIndex)
            {
                ai_mesh.mNormals[i] = mesh.VertexData<aiVector3D>(i, normal_attrib);
            }
            for (uint32_t j = 0; j < texcoord_attribs.size(); ++j)
            {
                if (texcoord_attribs[j] != VertexDesc::InvalidIndex)
                {
                    const float* texcoord = mesh.VertexDataPtr(i, texcoord_attribs[j]);
                    float* dst = &ai_mesh.mTextureCoords[j][i].x;
                    std::memcpy(dst, texcoord, std::min(ai_mesh.mNumUVComponents[j], 3U) * sizeof(float));
                    if (ai_mesh.mNumUVComponents[j] < 3)
                    {
                        std::memset(dst + ai_mesh.mNumUVComponents[j], 0, (3 - ai_mesh.mNumUVComponents[j]) * sizeof(float));
                    }
                }
            }
        }

        ai_mesh.mNumFaces = static_cast<uint32_t>(mesh.Indices().size() / 3);
        ai_mesh.mFaces = new aiFace[ai_mesh.mNumFaces];
        for (uint32_t j = 0; j < ai_mesh.mNumFaces; ++j)
        {
            auto& ai_face = ai_mesh.mFaces[j];
            ai_face.mIndices = new unsigned int[3];
            ai_face.mNumIndices = 3;

            ai_face.mIndices[0] = mesh.Index(j * 3 + 0);
            ai_face.mIndices[1] = mesh.Index(j * 3 + 1);
            ai_face.mIndices[2] = mesh.Index(j * 3 + 2);
        }

        ai_scene.mRootNode = new aiNode;
        ai_scene.mRootNode->mParent = nullptr;
        auto& ai_node = *ai_scene.mRootNode;
        ai_node.mName.Set("Root");

        ai_node.mTransformation = aiMatrix4x4();

        ai_node.mNumChildren = 0;
        ai_node.mChildren = nullptr;
        ai_node.mNumMeshes = 1;
        ai_node.mMeshes = new unsigned int[ai_node.mNumMeshes];
        ai_node.mMeshes[0] = 0;

        const char* format_id_table[][2] = {
            {"dae", "collada"},
            {"stl", "stlb"},
            {"ply", "plyb"},
            {"gltf", "gltf2"},
            {"glb", "glb2"},
        };

        auto format_id = ext.string().substr(1);
        for (size_t i = 0; i < std::size(format_id_table); ++i)
        {
            if (format_id == format_id_table[i][0])
            {
                format_id = format_id_table[i][1];
                break;
            }
        }

        Assimp::Exporter exporter;
        exporter.Export(&ai_scene, format_id.c_str(), path.string().c_str(), 0);
    }

    Mesh UnwrapUv(const Mesh& input_mesh, uint32_t texture_size, uint32_t padding, std::vector<uint32_t>* vertex_referencing)
    {
        Mesh ret_mesh;

        xatlas::Atlas* atlas = xatlas::Create();

        const auto& vertex_desc = input_mesh.MeshVertexDesc();
        const uint32_t pos_attrib = vertex_desc.FindAttrib(VertexAttrib::Semantic::Position, 0);

        xatlas::MeshDecl mesh_decl;
        mesh_decl.vertexCount = input_mesh.NumVertices();
        mesh_decl.vertexPositionData = input_mesh.VertexDataPtr(0, pos_attrib);
        mesh_decl.vertexPositionStride = vertex_desc.Stride();
        mesh_decl.indexCount = static_cast<uint32_t>(input_mesh.Indices().size());
        mesh_decl.indexData = input_mesh.Indices().data();
        mesh_decl.indexFormat = xatlas::IndexFormat::UInt32;

        xatlas::AddMeshError error = xatlas::AddMesh(atlas, mesh_decl, 1);
        if (error == xatlas::AddMeshError::Success)
        {
            xatlas::ChartOptions chart_options;

            xatlas::PackOptions pack_options;
            pack_options.padding = padding;
            pack_options.texelsPerUnit = 0;
            pack_options.resolution = texture_size;

            xatlas::Generate(atlas, chart_options, pack_options);

            assert(atlas->atlasCount == 1);
            assert(atlas->meshCount == 1);

            const xatlas::Mesh& mesh = atlas->meshes[0];

            std::vector<uint32_t> new_vertex_referencing(mesh.vertexCount);
            for (uint32_t vi = 0; vi < mesh.vertexCount; ++vi)
            {
                const auto& vertex = mesh.vertexArray[vi];
                new_vertex_referencing[vi] = vertex.xref;
            }

            VertexDesc new_vertex_desc;
            uint32_t texcoord_attrib = vertex_desc.FindAttrib(VertexAttrib::Semantic::TexCoord, 0);
            if (texcoord_attrib == VertexDesc::InvalidIndex)
            {
                const auto old_vertex_attrib = vertex_desc.Attribs();
                std::vector<VertexAttrib> new_vertex_attribs(old_vertex_attrib.begin(), old_vertex_attrib.end());
                new_vertex_attribs.push_back({VertexAttrib::Semantic::TexCoord, 0, 2});
                new_vertex_desc = VertexDesc(new_vertex_attribs);
                texcoord_attrib = new_vertex_desc.FindAttrib(VertexAttrib::Semantic::TexCoord, 0);
            }
            else
            {
                new_vertex_desc = vertex_desc;
            }

            ret_mesh =
                input_mesh.ExtractMesh(std::move(new_vertex_desc), new_vertex_referencing, std::span(mesh.indexArray, mesh.indexCount));
            for (uint32_t vi = 0; vi < mesh.vertexCount; ++vi)
            {
                const auto& vertex = mesh.vertexArray[vi];
                const XMFLOAT2 uv(vertex.uv[0] / atlas->width, vertex.uv[1] / atlas->height);
                ret_mesh.VertexData<XMFLOAT2>(vi, texcoord_attrib) = uv;
            }

            if (vertex_referencing)
            {
                (*vertex_referencing) = std::move(new_vertex_referencing);
            }
        }

        xatlas::Destroy(atlas);

        if (error != xatlas::AddMeshError::Success)
        {
            throw std::runtime_error(std::format("UV unwrapping failed {}", static_cast<uint32_t>(error)));
        }

        return ret_mesh;
    }
} // namespace AIHoloImager
