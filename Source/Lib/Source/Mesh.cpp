// Copyright (c) 2024 Minmin Gong
//

#include "AIHoloImager/Mesh.hpp"

#include <assimp/Exporter.hpp>
#include <assimp/GltfMaterial.h>
#include <assimp/Importer.hpp>
#include <assimp/material.h>
#include <assimp/scene.h>

using namespace DirectX;

namespace AIHoloImager
{
    class Mesh::Impl
    {
    public:
        Impl(uint32_t num_verts, uint32_t num_indices) : vertices_(num_verts), indices_(num_indices)
        {
        }

        std::span<const VertexFormat> Vertices() const noexcept
        {
            return std::span<const VertexFormat>(vertices_.begin(), vertices_.end());
        }
        void Vertices(std::span<const VertexFormat> verts)
        {
            vertices_.assign(verts.begin(), verts.end());
        }
        void ResizeVertices(uint32_t num)
        {
            vertices_.resize(num);
        }
        const VertexFormat& Vertex(uint32_t index) const
        {
            return vertices_[index];
        }
        VertexFormat& Vertex(uint32_t index)
        {
            return vertices_[index];
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

    private:
        std::vector<VertexFormat> vertices_;
        std::vector<uint32_t> indices_;

        Texture albedo_tex_;
    };

    Mesh::Mesh() = default;
    Mesh::Mesh(uint32_t num_verts, uint32_t num_indices) : impl_(std::make_unique<Impl>(num_verts, num_indices))
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

    std::span<const Mesh::VertexFormat> Mesh::Vertices() const noexcept
    {
        return impl_->Vertices();
    }
    void Mesh::Vertices(std::span<const Mesh::VertexFormat> verts)
    {
        impl_->Vertices(std::move(verts));
    }
    void Mesh::ResizeVertices(uint32_t num)
    {
        impl_->ResizeVertices(num);
    }
    const Mesh::VertexFormat& Mesh::Vertex(uint32_t index) const
    {
        return impl_->Vertex(index);
    }
    Mesh::VertexFormat& Mesh::Vertex(uint32_t index)
    {
        return impl_->Vertex(index);
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
            mesh = Mesh(ai_mesh->mNumVertices, ai_mesh->mNumFaces * 3);

            const aiMaterial* mtl = ai_scene->mMaterials[ai_mesh->mMaterialIndex];
            unsigned int count = aiGetMaterialTextureCount(mtl, aiTextureType_DIFFUSE);
            if (count > 0)
            {
                aiString str;
                aiGetMaterialTexture(mtl, aiTextureType_DIFFUSE, 0, &str, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
                mesh.AlbedoTexture() = LoadTexture(path.parent_path() / str.C_Str());
            }

            std::span<const Mesh::VertexFormat> vertices = mesh.Vertices();
            for (unsigned int vi = 0; vi < ai_mesh->mNumVertices; ++vi)
            {
                mesh.Vertex(vi) = {XMFLOAT3(&ai_mesh->mVertices[vi].x), XMFLOAT2(&ai_mesh->mTextureCoords[0][vi].x)};
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

        ai_mesh.mNumVertices = static_cast<uint32_t>(mesh.Vertices().size());

        ai_mesh.mVertices = new aiVector3D[ai_mesh.mNumVertices];
        ai_mesh.mTextureCoords[0] = new aiVector3D[ai_mesh.mNumVertices];
        ai_mesh.mNumUVComponents[0] = 2;

        for (uint32_t i = 0; i < ai_mesh.mNumVertices; ++i)
        {
            const auto& vertex = mesh.Vertex(i);
            ai_mesh.mVertices[i] = aiVector3D(vertex.pos.x, vertex.pos.y, vertex.pos.z);
            ai_mesh.mTextureCoords[0][i] = aiVector3D(vertex.texcoord.x, vertex.texcoord.y, 0);
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
} // namespace AIHoloImager
