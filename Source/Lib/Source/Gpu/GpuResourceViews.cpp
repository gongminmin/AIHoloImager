// Copyright (c) 2024 Minmin Gong
//

#include "GpuResourceViews.hpp"

#include "GpuSystem.hpp"
#include "GpuTexture.hpp"

namespace AIHoloImager
{
    GpuShaderResourceView::GpuShaderResourceView() noexcept = default;

    GpuShaderResourceView::GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture2D& texture)
        : GpuShaderResourceView(gpu_system, texture, DXGI_FORMAT_UNKNOWN)
    {
    }

    GpuShaderResourceView::GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture2D& texture, DXGI_FORMAT format)
        : GpuShaderResourceView(gpu_system, texture, ~0U, format)
    {
    }

    GpuShaderResourceView::GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture2D& texture, uint32_t sub_resource)
        : GpuShaderResourceView(gpu_system, texture, sub_resource, DXGI_FORMAT_UNKNOWN)
    {
    }

    GpuShaderResourceView::GpuShaderResourceView(
        GpuSystem& gpu_system, const GpuTexture2D& texture, uint32_t sub_resource, DXGI_FORMAT format)
        : gpu_system_(&gpu_system), texture_(&texture)
    {
        desc_block_ = gpu_system.AllocCbvSrvUavDescBlock(1);
        cpu_handle_ = desc_block_.CpuHandle();

        D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc{};
        srv_desc.Format = format == DXGI_FORMAT_UNKNOWN ? texture.Format() : format;
        srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        if (sub_resource == ~0U)
        {
            srv_desc.Texture2D.MostDetailedMip = 0;
            srv_desc.Texture2D.MipLevels = texture.MipLevels();
            srv_desc.Texture2D.PlaneSlice = 0;
        }
        else
        {
            SubResourceToMipLevelPlane(
                sub_resource, texture.MipLevels(), srv_desc.Texture2D.MostDetailedMip, srv_desc.Texture2D.PlaneSlice);
            srv_desc.Texture2D.MipLevels = 1;
        }
        srv_desc.Texture2D.ResourceMinLODClamp = 0;
        gpu_system.NativeDevice()->CreateShaderResourceView(texture.NativeTexture(), &srv_desc, cpu_handle_);
    }

    GpuShaderResourceView::GpuShaderResourceView(GpuSystem& gpu_system, const GpuBuffer& buffer, DXGI_FORMAT format)
        : GpuShaderResourceView(gpu_system, buffer, 0, buffer.Size() / FormatSize(format), format)
    {
    }

    GpuShaderResourceView::GpuShaderResourceView(
        GpuSystem& gpu_system, const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, DXGI_FORMAT format)
        : gpu_system_(&gpu_system), buffer_(&buffer)
    {
        desc_block_ = gpu_system.AllocCbvSrvUavDescBlock(1);
        cpu_handle_ = desc_block_.CpuHandle();

        D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc{};
        srv_desc.Format = format;
        srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv_desc.Buffer.FirstElement = first_element;
        srv_desc.Buffer.NumElements = num_elements;
        srv_desc.Buffer.StructureByteStride = 0;
        srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
        gpu_system.NativeDevice()->CreateShaderResourceView(buffer.NativeBuffer(), &srv_desc, cpu_handle_);
    }

    GpuShaderResourceView::~GpuShaderResourceView()
    {
        this->Reset();
    }

    GpuShaderResourceView::GpuShaderResourceView(GpuShaderResourceView&& other) noexcept = default;
    GpuShaderResourceView& GpuShaderResourceView::operator=(GpuShaderResourceView&& other) noexcept = default;

    void GpuShaderResourceView::Reset()
    {
        cpu_handle_ = {};
        if (desc_block_)
        {
            gpu_system_->DeallocCbvSrvUavDescBlock(std::move(desc_block_));
        }
    }

    void GpuShaderResourceView::Transition(GpuCommandList& cmd_list) const
    {
        if (texture_ != nullptr)
        {
            texture_->Transition(cmd_list, D3D12_RESOURCE_STATE_COMMON);
        }
        else if (buffer_ != nullptr)
        {
            buffer_->Transition(cmd_list, D3D12_RESOURCE_STATE_COMMON);
        }
    }

    void GpuShaderResourceView::CopyTo(D3D12_CPU_DESCRIPTOR_HANDLE dst_handle) const noexcept
    {
        gpu_system_->NativeDevice()->CopyDescriptorsSimple(1, dst_handle, cpu_handle_, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }


    GpuRenderTargetView::GpuRenderTargetView() noexcept = default;

    GpuRenderTargetView::GpuRenderTargetView(GpuSystem& gpu_system, GpuTexture2D& texture)
        : GpuRenderTargetView(gpu_system, texture, DXGI_FORMAT_UNKNOWN)
    {
    }
    GpuRenderTargetView::GpuRenderTargetView(GpuSystem& gpu_system, GpuTexture2D& texture, DXGI_FORMAT format)
        : gpu_system_(&gpu_system), texture_(&texture)
    {
        desc_block_ = gpu_system.AllocRtvDescBlock(1);
        cpu_handle_ = desc_block_.CpuHandle();

        D3D12_RENDER_TARGET_VIEW_DESC rtv_desc{};
        rtv_desc.Format = (format == DXGI_FORMAT_UNKNOWN) ? texture.Format() : format;
        rtv_desc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        gpu_system.NativeDevice()->CreateRenderTargetView(texture.NativeTexture(), &rtv_desc, cpu_handle_);
    }

    GpuRenderTargetView::~GpuRenderTargetView()
    {
        this->Reset();
    }

    GpuRenderTargetView::GpuRenderTargetView(GpuRenderTargetView&& other) noexcept = default;
    GpuRenderTargetView& GpuRenderTargetView::operator=(GpuRenderTargetView&& other) noexcept = default;

    GpuRenderTargetView::operator bool() const noexcept
    {
        return (cpu_handle_.ptr != 0);
    }

    void GpuRenderTargetView::Reset()
    {
        cpu_handle_ = {};
        if (desc_block_)
        {
            gpu_system_->DeallocRtvDescBlock(std::move(desc_block_));
        }
    }

    void GpuRenderTargetView::Transition(GpuCommandList& cmd_list) const
    {
        texture_->Transition(cmd_list, D3D12_RESOURCE_STATE_RENDER_TARGET);
    }

    D3D12_CPU_DESCRIPTOR_HANDLE GpuRenderTargetView::CpuHandle() const noexcept
    {
        return cpu_handle_;
    }


    GpuDepthStencilView::GpuDepthStencilView() noexcept = default;

    GpuDepthStencilView::GpuDepthStencilView(GpuSystem& gpu_system, GpuTexture2D& texture)
        : GpuDepthStencilView(gpu_system, texture, DXGI_FORMAT_UNKNOWN)
    {
    }
    GpuDepthStencilView::GpuDepthStencilView(GpuSystem& gpu_system, GpuTexture2D& texture, DXGI_FORMAT format)
        : gpu_system_(&gpu_system), texture_(&texture)
    {
        desc_block_ = gpu_system.AllocDsvDescBlock(1);
        cpu_handle_ = desc_block_.CpuHandle();

        D3D12_DEPTH_STENCIL_VIEW_DESC dsv_desc{};
        dsv_desc.Format = (format == DXGI_FORMAT_UNKNOWN) ? texture.Format() : format;
        dsv_desc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
        gpu_system.NativeDevice()->CreateDepthStencilView(texture.NativeTexture(), &dsv_desc, cpu_handle_);
    }

    GpuDepthStencilView::~GpuDepthStencilView()
    {
        this->Reset();
    }

    GpuDepthStencilView::GpuDepthStencilView(GpuDepthStencilView&& other) noexcept = default;
    GpuDepthStencilView& GpuDepthStencilView::operator=(GpuDepthStencilView&& other) noexcept = default;

    GpuDepthStencilView::operator bool() const noexcept
    {
        return (cpu_handle_.ptr != 0);
    }

    void GpuDepthStencilView::Reset()
    {
        cpu_handle_ = {};
        if (desc_block_)
        {
            gpu_system_->DeallocDsvDescBlock(std::move(desc_block_));
        }
    }

    void GpuDepthStencilView::Transition(GpuCommandList& cmd_list) const
    {
        texture_->Transition(cmd_list, D3D12_RESOURCE_STATE_DEPTH_WRITE);
    }

    D3D12_CPU_DESCRIPTOR_HANDLE GpuDepthStencilView::CpuHandle() const noexcept
    {
        return cpu_handle_;
    }


    GpuUnorderedAccessView::GpuUnorderedAccessView() noexcept = default;

    GpuUnorderedAccessView::GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2D& texture)
        : GpuUnorderedAccessView(gpu_system, texture, DXGI_FORMAT_UNKNOWN)
    {
    }

    GpuUnorderedAccessView::GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2D& texture, DXGI_FORMAT format)
        : GpuUnorderedAccessView(gpu_system, texture, 0, format)
    {
    }

    GpuUnorderedAccessView::GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2D& texture, uint32_t sub_resource)
        : GpuUnorderedAccessView(gpu_system, texture, sub_resource, DXGI_FORMAT_UNKNOWN)
    {
    }

    GpuUnorderedAccessView::GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2D& texture, uint32_t sub_resource, DXGI_FORMAT format)
        : gpu_system_(&gpu_system), texture_(&texture)
    {
        desc_block_ = gpu_system.AllocCbvSrvUavDescBlock(1);
        cpu_handle_ = desc_block_.CpuHandle();

        D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc{};
        uav_desc.Format = format == DXGI_FORMAT_UNKNOWN ? texture.Format() : format;
        uav_desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        SubResourceToMipLevelPlane(sub_resource, texture.MipLevels(), uav_desc.Texture2D.MipSlice, uav_desc.Texture2D.PlaneSlice);
        gpu_system.NativeDevice()->CreateUnorderedAccessView(texture.NativeTexture(), nullptr, &uav_desc, cpu_handle_);
    }

    GpuUnorderedAccessView::GpuUnorderedAccessView(GpuSystem& gpu_system, GpuBuffer& buffer, uint32_t element_size)
        : GpuUnorderedAccessView(gpu_system, buffer, 0, buffer.Size() / element_size, element_size)
    {
    }

    GpuUnorderedAccessView::GpuUnorderedAccessView(
        GpuSystem& gpu_system, GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size)
        : gpu_system_(&gpu_system), buffer_(&buffer)
    {
        desc_block_ = gpu_system.AllocCbvSrvUavDescBlock(1);
        cpu_handle_ = desc_block_.CpuHandle();

        D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc{};
        uav_desc.Format = DXGI_FORMAT_UNKNOWN;
        uav_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uav_desc.Buffer.FirstElement = first_element;
        uav_desc.Buffer.NumElements = num_elements;
        uav_desc.Buffer.StructureByteStride = element_size;
        gpu_system.NativeDevice()->CreateUnorderedAccessView(buffer.NativeBuffer(), nullptr, &uav_desc, cpu_handle_);
    }

    GpuUnorderedAccessView::~GpuUnorderedAccessView()
    {
        this->Reset();
    }

    GpuUnorderedAccessView::GpuUnorderedAccessView(GpuUnorderedAccessView&& other) noexcept = default;
    GpuUnorderedAccessView& GpuUnorderedAccessView::operator=(GpuUnorderedAccessView&& other) noexcept = default;

    void GpuUnorderedAccessView::Reset()
    {
        cpu_handle_ = {};
        if (desc_block_)
        {
            gpu_system_->DeallocCbvSrvUavDescBlock(std::move(desc_block_));
        }
    }

    void GpuUnorderedAccessView::Transition(GpuCommandList& cmd_list) const
    {
        if (texture_ != nullptr)
        {
            texture_->Transition(cmd_list, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        else if (buffer_ != nullptr)
        {
            buffer_->Transition(cmd_list, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
    }

    void GpuUnorderedAccessView::CopyTo(D3D12_CPU_DESCRIPTOR_HANDLE dst_handle) const noexcept
    {
        gpu_system_->NativeDevice()->CopyDescriptorsSimple(1, dst_handle, cpu_handle_, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }
} // namespace AIHoloImager
