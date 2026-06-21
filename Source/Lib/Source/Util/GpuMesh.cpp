// Copyright (c) 2026 Minmin Gong
//

#include "GpuMesh.hpp"

#include <cassert>
#include <future>

#include "Base/ErrorHandling.hpp"
#include "FormatConversion.hpp"
#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuSystem.hpp"

namespace AIHoloImager
{
    class GpuMesh::Impl
    {
    public:
        Impl(GpuVertexLayout vertex_desc, GpuFormat index_format) : vertex_desc_(std::move(vertex_desc)), index_format_(index_format)
        {
        }

        const GpuVertexLayout& MeshVertexDesc() const noexcept
        {
            return vertex_desc_;
        }
        GpuFormat IndexFormat() const noexcept
        {
            return index_format_;
        }
        GpuBuffer& VertexBuffer() noexcept
        {
            return vb_;
        }
        GpuBuffer& IndexBuffer() noexcept
        {
            return ib_;
        }

        GpuTexture2D& AlbedoTexture() noexcept
        {
            return albedo_;
        }

        uint32_t NumVertices() const noexcept
        {
            return vb_ ? vb_.Size() / vertex_desc_.SlotStrides()[0] : 0;
        }
        uint32_t NumIndices() const noexcept
        {
            return ib_ ? ib_.Size() / FormatSize(index_format_) : 0;
        }

    private:
        GpuVertexLayout vertex_desc_;
        GpuFormat index_format_;
        GpuBuffer vb_;
        GpuBuffer ib_;
        GpuTexture2D albedo_;
    };

    GpuMesh::GpuMesh() noexcept = default;
    GpuMesh::GpuMesh(GpuVertexLayout vertex_desc, GpuFormat index_format)
        : impl_(std::make_unique<Impl>(std::move(vertex_desc), index_format))
    {
    }
    GpuMesh::~GpuMesh() noexcept = default;

    GpuMesh::GpuMesh(GpuMesh&& rhs) noexcept = default;
    GpuMesh& GpuMesh::operator=(GpuMesh&& rhs) noexcept = default;

    bool GpuMesh::Valid() const noexcept
    {
        return static_cast<bool>(impl_);
    }

    const GpuVertexLayout& GpuMesh::MeshVertexDesc() const noexcept
    {
        assert(impl_);
        return impl_->MeshVertexDesc();
    }
    GpuFormat GpuMesh::IndexFormat() const noexcept
    {
        assert(impl_);
        return impl_->IndexFormat();
    }
    GpuBuffer& GpuMesh::VertexBuffer() noexcept
    {
        assert(impl_);
        return impl_->VertexBuffer();
    }
    const GpuBuffer& GpuMesh::VertexBuffer() const noexcept
    {
        return const_cast<GpuMesh*>(this)->VertexBuffer();
    }
    GpuBuffer& GpuMesh::IndexBuffer() noexcept
    {
        assert(impl_);
        return impl_->IndexBuffer();
    }
    const GpuBuffer& GpuMesh::IndexBuffer() const noexcept
    {
        return const_cast<GpuMesh*>(this)->IndexBuffer();
    }

    GpuTexture2D& GpuMesh::AlbedoTexture() noexcept
    {
        assert(impl_);
        return impl_->AlbedoTexture();
    }
    const GpuTexture2D& GpuMesh::AlbedoTexture() const noexcept
    {
        return const_cast<GpuMesh*>(this)->AlbedoTexture();
    }

    uint32_t GpuMesh::NumVertices() const noexcept
    {
        assert(impl_);
        return impl_->NumVertices();
    }
    uint32_t GpuMesh::NumIndices() const noexcept
    {
        assert(impl_);
        return impl_->NumIndices();
    }

    Mesh ToMesh(GpuSystem& gpu_system, const GpuMesh& gpu_mesh)
    {
        const auto& gpu_vertex_layout = gpu_mesh.MeshVertexDesc();
        const auto gpu_vertex_attribs = gpu_vertex_layout.Attribs();
        auto vertex_attribs = std::make_unique<VertexAttrib[]>(gpu_vertex_attribs.size());
        for (size_t i = 0; i < gpu_vertex_attribs.size(); ++i)
        {
            if (gpu_vertex_attribs[i].semantic == "POSITION")
            {
                vertex_attribs[i].semantic = VertexAttrib::Semantic::Position;
            }
            else if (gpu_vertex_attribs[i].semantic == "TEXCOORD")
            {
                vertex_attribs[i].semantic = VertexAttrib::Semantic::TexCoord;
            }
            else if (gpu_vertex_attribs[i].semantic == "NORMAL")
            {
                vertex_attribs[i].semantic = VertexAttrib::Semantic::Normal;
            }
            else if (gpu_vertex_attribs[i].semantic == "COLOR")
            {
                vertex_attribs[i].semantic = VertexAttrib::Semantic::Color;
            }
            else
            {
                Unreachable("Invalid vertex attribute semantic");
            }
            vertex_attribs[i].index = gpu_vertex_attribs[i].semantic_index;

            assert(BaseFormat(gpu_vertex_attribs[i].format) == GpuBaseFormat::Float); // Only support float vertex attributes for now
            vertex_attribs[i].channels = FormatChannels(gpu_vertex_attribs[i].format);
            vertex_attribs[i].offset = VertexAttrib::AppendOffset;
        }

        assert(gpu_mesh.IndexFormat() == GpuFormat::R32_Uint); // Only support uint32_t index format for now
        Mesh ret(VertexDesc(std::span(vertex_attribs.get(), gpu_vertex_attribs.size())), gpu_mesh.NumVertices(), gpu_mesh.NumIndices());

        auto cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Copy);

        std::future<void> vertex_rb_future =
            cmd_list.ReadBackAsync(gpu_mesh.VertexBuffer(), ret.VertexBuffer().data(), ret.VertexBuffer().size_bytes());
        std::future<void> index_rb_future =
            cmd_list.ReadBackAsync(gpu_mesh.IndexBuffer(), ret.IndexBuffer().data(), ret.IndexBuffer().size_bytes());

        const auto& albedo_tex = gpu_mesh.AlbedoTexture();
        std::future<void> albedo_rb_future;
        if (albedo_tex)
        {
            Texture albedo(albedo_tex.Width(0), albedo_tex.Height(0), ToElementFormat(albedo_tex.Format()));
            albedo_rb_future = cmd_list.ReadBackAsync(albedo_tex, 0, albedo.Data(), albedo.DataSize());
            ret.AlbedoTexture() = std::move(albedo);
        }

        gpu_system.Execute(std::move(cmd_list));

        vertex_rb_future.wait();
        index_rb_future.wait();
        if (albedo_tex)
        {
            albedo_rb_future.wait();
        }

        return ret;
    }

    GpuMesh ToGpuMesh(GpuSystem& gpu_system, const Mesh& mesh)
    {
        const auto& vertex_desc = mesh.MeshVertexDesc();
        const auto vertex_attribs = vertex_desc.Attribs();
        auto gpu_vertex_attribs = std::make_unique<GpuVertexAttrib[]>(vertex_attribs.size());
        for (size_t i = 0; i < vertex_attribs.size(); ++i)
        {
            switch (vertex_attribs[i].semantic)
            {
            case VertexAttrib::Semantic::Position:
                gpu_vertex_attribs[i].semantic = "POSITION";
                break;
            case VertexAttrib::Semantic::TexCoord:
                gpu_vertex_attribs[i].semantic = "TEXCOORD";
                break;
            case VertexAttrib::Semantic::Normal:
                gpu_vertex_attribs[i].semantic = "NORMAL";
                break;
            case VertexAttrib::Semantic::Color:
                gpu_vertex_attribs[i].semantic = "COLOR";
                break;

            default:
                Unreachable("Invalid vertex attribute semantic");
            }
            gpu_vertex_attribs[i].semantic_index = vertex_attribs[i].index;

            switch (vertex_attribs[i].channels)
            {
            case 1:
                gpu_vertex_attribs[i].format = GpuFormat::R32_Float;
                break;
            case 2:
                gpu_vertex_attribs[i].format = GpuFormat::RG32_Float;
                break;
            case 3:
                gpu_vertex_attribs[i].format = GpuFormat::RGB32_Float;
                break;
            case 4:
                gpu_vertex_attribs[i].format = GpuFormat::RGBA32_Float;
                break;

            default:
                Unreachable("Invalid vertex attribute channels");
            }

            gpu_vertex_attribs[i].offset = vertex_attribs[i].offset;
        }

        const auto cpu_vb = mesh.VertexBuffer();
        const auto cpu_ib = mesh.IndexBuffer();
        const auto& cpu_albedo = mesh.AlbedoTexture();

        GpuMesh ret(GpuVertexLayout(gpu_system, std::span(gpu_vertex_attribs.get(), vertex_attribs.size())), GpuFormat::R32_Uint);

        GpuBuffer vb(gpu_system, static_cast<uint32_t>(cpu_vb.size() * sizeof(float)), GpuHeap::Default,
            GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess | GpuResourceFlag::VertexBuffer | GpuResourceFlag::Shareable,
            "vb");
        GpuBuffer ib(gpu_system, static_cast<uint32_t>(cpu_ib.size() * sizeof(uint32_t)), GpuHeap::Default,
            GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess | GpuResourceFlag::IndexBuffer | GpuResourceFlag::Shareable,
            "ib");

        GpuTexture2D albedo_tex;
        if (cpu_albedo.Valid())
        {
            albedo_tex = GpuTexture2D(gpu_system, cpu_albedo.Width(), cpu_albedo.Height(), 1, ToGpuFormat(cpu_albedo.Format()),
                GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess, "albedo");
        }

        GpuCommandList cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Copy);

        cmd_list.Upload(vb, cpu_vb.data(), vb.Size());
        cmd_list.Upload(ib, cpu_ib.data(), ib.Size());
        ret.VertexBuffer() = std::move(vb);
        ret.IndexBuffer() = std::move(ib);

        if (cpu_albedo.Valid())
        {
            cmd_list.Upload(albedo_tex, 0, cpu_albedo.Data(), cpu_albedo.DataSize());
            ret.AlbedoTexture() = std::move(albedo_tex);
        }

        gpu_system.Execute(std::move(cmd_list));

        return ret;
    }

    GpuMesh CopyGpuMesh(GpuSystem& gpu_system, const GpuMesh& mesh)
    {
        GpuMesh ret(mesh.MeshVertexDesc(), mesh.IndexFormat());

        const auto& src_vb = mesh.VertexBuffer();
        const auto& src_ib = mesh.IndexBuffer();
        const auto& src_albedo = mesh.AlbedoTexture();

        GpuBuffer dst_vb(gpu_system, mesh.VertexBuffer().Size(), GpuHeap::Default,
            GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess | GpuResourceFlag::VertexBuffer | GpuResourceFlag::Shareable,
            "vb");
        GpuBuffer dst_ib(gpu_system, mesh.IndexBuffer().Size(), GpuHeap::Default,
            GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess | GpuResourceFlag::IndexBuffer | GpuResourceFlag::Shareable,
            "ib");

        GpuTexture2D dst_albedo;
        if (src_albedo)
        {
            dst_albedo = GpuTexture2D(gpu_system, src_albedo.Width(0), src_albedo.Height(0), 1, src_albedo.Format(),
                GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess, "albedo");
        }

        GpuCommandList cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Copy);

        cmd_list.Copy(dst_vb, src_vb);
        cmd_list.Copy(dst_ib, src_ib);
        ret.VertexBuffer() = std::move(dst_vb);
        ret.IndexBuffer() = std::move(dst_ib);

        if (src_albedo)
        {
            cmd_list.Copy(dst_albedo, src_albedo);
            ret.AlbedoTexture() = std::move(dst_albedo);
        }

        gpu_system.Execute(std::move(cmd_list));

        return ret;
    }
} // namespace AIHoloImager
