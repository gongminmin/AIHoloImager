// Copyright (c) 2024-2025 Minmin Gong
//

#include "Gpu/GpuResourceViews.hpp"

#include "Gpu/D3D12/D3D12Traits.hpp"
#include "Gpu/GpuFormat.hpp"
#include "Gpu/GpuSystem.hpp"
#include "Gpu/GpuTexture.hpp"

#include "D3D12/D3D12Conversion.hpp"

namespace AIHoloImager
{
    GpuShaderResourceView::GpuShaderResourceView() noexcept = default;

    GpuShaderResourceView::GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture2D& texture)
        : GpuShaderResourceView(gpu_system, texture, GpuFormat::Unknown)
    {
    }

    GpuShaderResourceView::GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture2D& texture, GpuFormat format)
        : GpuShaderResourceView(gpu_system, texture, ~0U, format)
    {
    }

    GpuShaderResourceView::GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture2D& texture, uint32_t sub_resource)
        : GpuShaderResourceView(gpu_system, texture, sub_resource, GpuFormat::Unknown)
    {
    }

    GpuShaderResourceView::GpuShaderResourceView(
        GpuSystem& gpu_system, const GpuTexture2D& texture, uint32_t sub_resource, GpuFormat format)
        : gpu_system_(&gpu_system), texture_2d_(&texture)
    {
        desc_block_ = gpu_system.AllocCbvSrvUavDescBlock(1);
        cpu_handle_ = desc_block_.CpuHandle();

        D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc{};
        srv_desc.Format = ToDxgiFormat(format == GpuFormat::Unknown ? texture.Format() : format);
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
            uint32_t array_slice;
            DecomposeSubResource(
                sub_resource, texture.MipLevels(), 1, srv_desc.Texture2D.MostDetailedMip, array_slice, srv_desc.Texture2D.PlaneSlice);
            srv_desc.Texture2D.MipLevels = 1;
        }
        srv_desc.Texture2D.ResourceMinLODClamp = 0;
        gpu_system.NativeDevice()->CreateShaderResourceView(texture.NativeTexture<D3D12Traits>(), &srv_desc, cpu_handle_);
    }

    GpuShaderResourceView::GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture2DArray& texture_array)
        : GpuShaderResourceView(gpu_system, texture_array, GpuFormat::Unknown)
    {
    }

    GpuShaderResourceView::GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture2DArray& texture_array, GpuFormat format)
        : GpuShaderResourceView(gpu_system, texture_array, ~0U, format)
    {
    }

    GpuShaderResourceView::GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture2DArray& texture_array, uint32_t sub_resource)
        : GpuShaderResourceView(gpu_system, texture_array, sub_resource, GpuFormat::Unknown)
    {
    }

    GpuShaderResourceView::GpuShaderResourceView(
        GpuSystem& gpu_system, const GpuTexture2DArray& texture_array, uint32_t sub_resource, GpuFormat format)
        : gpu_system_(&gpu_system), texture_2d_array_(&texture_array)
    {
        desc_block_ = gpu_system.AllocCbvSrvUavDescBlock(1);
        cpu_handle_ = desc_block_.CpuHandle();

        D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc{};
        srv_desc.Format = ToDxgiFormat(format == GpuFormat::Unknown ? texture_array.Format() : format);
        srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        if (sub_resource == ~0U)
        {
            srv_desc.Texture2DArray.MostDetailedMip = 0;
            srv_desc.Texture2DArray.MipLevels = texture_array.MipLevels();
            srv_desc.Texture2DArray.PlaneSlice = 0;
            srv_desc.Texture2DArray.ArraySize = texture_array.ArraySize();
            srv_desc.Texture2DArray.FirstArraySlice = 0;
        }
        else
        {
            DecomposeSubResource(sub_resource, texture_array.MipLevels(), texture_array.ArraySize(),
                srv_desc.Texture2DArray.MostDetailedMip, srv_desc.Texture2DArray.FirstArraySlice, srv_desc.Texture2DArray.PlaneSlice);
            srv_desc.Texture2DArray.MipLevels = 1;
        }
        srv_desc.Texture2DArray.ResourceMinLODClamp = 0;
        gpu_system.NativeDevice()->CreateShaderResourceView(texture_array.NativeTexture<D3D12Traits>(), &srv_desc, cpu_handle_);
    }

    GpuShaderResourceView::GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture3D& texture)
        : GpuShaderResourceView(gpu_system, texture, GpuFormat::Unknown)
    {
    }

    GpuShaderResourceView::GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture3D& texture, GpuFormat format)
        : GpuShaderResourceView(gpu_system, texture, ~0U, format)
    {
    }

    GpuShaderResourceView::GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture3D& texture, uint32_t sub_resource)
        : GpuShaderResourceView(gpu_system, texture, sub_resource, GpuFormat::Unknown)
    {
    }

    GpuShaderResourceView::GpuShaderResourceView(
        GpuSystem& gpu_system, const GpuTexture3D& texture, uint32_t sub_resource, GpuFormat format)
        : gpu_system_(&gpu_system), texture_3d_(&texture)
    {
        desc_block_ = gpu_system.AllocCbvSrvUavDescBlock(1);
        cpu_handle_ = desc_block_.CpuHandle();

        D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc{};
        srv_desc.Format = ToDxgiFormat(format == GpuFormat::Unknown ? texture.Format() : format);
        srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
        srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        if (sub_resource == ~0U)
        {
            srv_desc.Texture3D.MostDetailedMip = 0;
            srv_desc.Texture3D.MipLevels = texture.MipLevels();
        }
        else
        {
            uint32_t array_slice;
            uint32_t plane_slice;
            DecomposeSubResource(sub_resource, texture.MipLevels(), 1, srv_desc.Texture3D.MostDetailedMip, array_slice, plane_slice);
            srv_desc.Texture3D.MipLevels = 1;
        }
        srv_desc.Texture3D.ResourceMinLODClamp = 0;
        gpu_system.NativeDevice()->CreateShaderResourceView(texture.NativeTexture<D3D12Traits>(), &srv_desc, cpu_handle_);
    }

    GpuShaderResourceView::GpuShaderResourceView(GpuSystem& gpu_system, const GpuBuffer& buffer, GpuFormat format)
        : GpuShaderResourceView(gpu_system, buffer, 0, buffer.Size() / FormatSize(format), format)
    {
    }

    GpuShaderResourceView::GpuShaderResourceView(
        GpuSystem& gpu_system, const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, GpuFormat format)
        : gpu_system_(&gpu_system), buffer_(&buffer)
    {
        desc_block_ = gpu_system.AllocCbvSrvUavDescBlock(1);
        cpu_handle_ = desc_block_.CpuHandle();

        D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc{};
        srv_desc.Format = ToDxgiFormat(format);
        srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv_desc.Buffer.FirstElement = first_element;
        srv_desc.Buffer.NumElements = num_elements;
        srv_desc.Buffer.StructureByteStride = 0;
        srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
        gpu_system.NativeDevice()->CreateShaderResourceView(buffer.NativeBuffer<D3D12Traits>(), &srv_desc, cpu_handle_);
    }

    GpuShaderResourceView::GpuShaderResourceView(GpuSystem& gpu_system, const GpuBuffer& buffer, uint32_t element_size)
        : GpuShaderResourceView(gpu_system, buffer, 0, buffer.Size() / element_size, element_size)
    {
    }

    GpuShaderResourceView::GpuShaderResourceView(
        GpuSystem& gpu_system, const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size)
        : gpu_system_(&gpu_system), buffer_(&buffer)
    {
        desc_block_ = gpu_system.AllocCbvSrvUavDescBlock(1);
        cpu_handle_ = desc_block_.CpuHandle();

        D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc{};
        srv_desc.Format = DXGI_FORMAT_UNKNOWN;
        srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv_desc.Buffer.FirstElement = first_element;
        srv_desc.Buffer.NumElements = num_elements;
        srv_desc.Buffer.StructureByteStride = element_size;
        srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
        gpu_system.NativeDevice()->CreateShaderResourceView(buffer.NativeBuffer<D3D12Traits>(), &srv_desc, cpu_handle_);
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
        if (texture_2d_ != nullptr)
        {
            texture_2d_->Transition(cmd_list, GpuResourceState::Common);
        }
        else if (texture_2d_array_ != nullptr)
        {
            texture_2d_array_->Transition(cmd_list, GpuResourceState::Common);
        }
        else if (texture_3d_ != nullptr)
        {
            texture_3d_->Transition(cmd_list, GpuResourceState::Common);
        }
        else if (buffer_ != nullptr)
        {
            buffer_->Transition(cmd_list, GpuResourceState::Common);
        }
    }

    void GpuShaderResourceView::CopyTo(D3D12_CPU_DESCRIPTOR_HANDLE dst_handle) const noexcept
    {
        gpu_system_->NativeDevice()->CopyDescriptorsSimple(1, dst_handle, cpu_handle_, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }


    GpuRenderTargetView::GpuRenderTargetView() noexcept = default;

    GpuRenderTargetView::GpuRenderTargetView(GpuSystem& gpu_system, GpuTexture2D& texture)
        : GpuRenderTargetView(gpu_system, texture, GpuFormat::Unknown)
    {
    }
    GpuRenderTargetView::GpuRenderTargetView(GpuSystem& gpu_system, GpuTexture2D& texture, GpuFormat format)
        : gpu_system_(&gpu_system), texture_(&texture)
    {
        desc_block_ = gpu_system.AllocRtvDescBlock(1);
        cpu_handle_ = desc_block_.CpuHandle();

        D3D12_RENDER_TARGET_VIEW_DESC rtv_desc{};
        rtv_desc.Format = ToDxgiFormat(format == GpuFormat::Unknown ? texture.Format() : format);
        rtv_desc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        gpu_system.NativeDevice()->CreateRenderTargetView(texture.NativeTexture<D3D12Traits>(), &rtv_desc, cpu_handle_);
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
        texture_->Transition(cmd_list, GpuResourceState::ColorWrite);
    }

    D3D12_CPU_DESCRIPTOR_HANDLE GpuRenderTargetView::CpuHandle() const noexcept
    {
        return cpu_handle_;
    }


    GpuDepthStencilView::GpuDepthStencilView() noexcept = default;

    GpuDepthStencilView::GpuDepthStencilView(GpuSystem& gpu_system, GpuTexture2D& texture)
        : GpuDepthStencilView(gpu_system, texture, GpuFormat::Unknown)
    {
    }
    GpuDepthStencilView::GpuDepthStencilView(GpuSystem& gpu_system, GpuTexture2D& texture, GpuFormat format)
        : gpu_system_(&gpu_system), texture_(&texture)
    {
        desc_block_ = gpu_system.AllocDsvDescBlock(1);
        cpu_handle_ = desc_block_.CpuHandle();

        D3D12_DEPTH_STENCIL_VIEW_DESC dsv_desc{};
        dsv_desc.Format = ToDxgiFormat(format == GpuFormat::Unknown ? texture.Format() : format);
        dsv_desc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
        gpu_system.NativeDevice()->CreateDepthStencilView(texture.NativeTexture<D3D12Traits>(), &dsv_desc, cpu_handle_);
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
        texture_->Transition(cmd_list, GpuResourceState::DepthWrite);
    }

    D3D12_CPU_DESCRIPTOR_HANDLE GpuDepthStencilView::CpuHandle() const noexcept
    {
        return cpu_handle_;
    }


    GpuUnorderedAccessView::GpuUnorderedAccessView() noexcept = default;

    GpuUnorderedAccessView::GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2D& texture)
        : GpuUnorderedAccessView(gpu_system, texture, GpuFormat::Unknown)
    {
    }

    GpuUnorderedAccessView::GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2D& texture, GpuFormat format)
        : GpuUnorderedAccessView(gpu_system, texture, 0, format)
    {
    }

    GpuUnorderedAccessView::GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2D& texture, uint32_t sub_resource)
        : GpuUnorderedAccessView(gpu_system, texture, sub_resource, GpuFormat::Unknown)
    {
    }

    GpuUnorderedAccessView::GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2D& texture, uint32_t sub_resource, GpuFormat format)
        : gpu_system_(&gpu_system), texture_2d_(&texture)
    {
        desc_block_ = gpu_system.AllocCbvSrvUavDescBlock(1);
        cpu_handle_ = desc_block_.CpuHandle();

        D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc{};
        uav_desc.Format = ToDxgiFormat(format == GpuFormat::Unknown ? texture.Format() : format);
        uav_desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        uint32_t array_slice;
        DecomposeSubResource(sub_resource, texture.MipLevels(), 1, uav_desc.Texture2D.MipSlice, array_slice, uav_desc.Texture2D.PlaneSlice);
        gpu_system.NativeDevice()->CreateUnorderedAccessView(texture.NativeTexture<D3D12Traits>(), nullptr, &uav_desc, cpu_handle_);
    }

    GpuUnorderedAccessView::GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2DArray& texture_array)
        : GpuUnorderedAccessView(gpu_system, texture_array, GpuFormat::Unknown)
    {
    }

    GpuUnorderedAccessView::GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2DArray& texture_array, GpuFormat format)
        : GpuUnorderedAccessView(gpu_system, texture_array, 0, format)
    {
    }

    GpuUnorderedAccessView::GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2DArray& texture_array, uint32_t sub_resource)
        : GpuUnorderedAccessView(gpu_system, texture_array, sub_resource, GpuFormat::Unknown)
    {
    }

    GpuUnorderedAccessView::GpuUnorderedAccessView(
        GpuSystem& gpu_system, GpuTexture2DArray& texture_array, uint32_t sub_resource, GpuFormat format)
        : gpu_system_(&gpu_system), texture_2d_array_(&texture_array)
    {
        desc_block_ = gpu_system.AllocCbvSrvUavDescBlock(1);
        cpu_handle_ = desc_block_.CpuHandle();

        D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc{};
        uav_desc.Format = ToDxgiFormat(format == GpuFormat::Unknown ? texture_array.Format() : format);
        uav_desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        uav_desc.Texture2DArray.ArraySize = 1;
        DecomposeSubResource(sub_resource, texture_array.MipLevels(), texture_array.ArraySize(), uav_desc.Texture2DArray.MipSlice,
            uav_desc.Texture2DArray.FirstArraySlice, uav_desc.Texture2DArray.PlaneSlice);
        gpu_system.NativeDevice()->CreateUnorderedAccessView(texture_array.NativeTexture<D3D12Traits>(), nullptr, &uav_desc, cpu_handle_);
    }

    GpuUnorderedAccessView::GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture3D& texture)
        : GpuUnorderedAccessView(gpu_system, texture, GpuFormat::Unknown)
    {
    }

    GpuUnorderedAccessView::GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture3D& texture, GpuFormat format)
        : GpuUnorderedAccessView(gpu_system, texture, 0, format)
    {
    }

    GpuUnorderedAccessView::GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture3D& texture, uint32_t sub_resource)
        : GpuUnorderedAccessView(gpu_system, texture, sub_resource, GpuFormat::Unknown)
    {
    }

    GpuUnorderedAccessView::GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture3D& texture, uint32_t sub_resource, GpuFormat format)
        : gpu_system_(&gpu_system), texture_3d_(&texture)
    {
        desc_block_ = gpu_system.AllocCbvSrvUavDescBlock(1);
        cpu_handle_ = desc_block_.CpuHandle();

        D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc{};
        uav_desc.Format = ToDxgiFormat(format == GpuFormat::Unknown ? texture.Format() : format);
        uav_desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
        uint32_t array_slice;
        uint32_t plane_slice;
        DecomposeSubResource(sub_resource, texture.MipLevels(), 1, uav_desc.Texture3D.MipSlice, array_slice, plane_slice);
        uav_desc.Texture3D.FirstWSlice = 0;
        uav_desc.Texture3D.WSize = texture.Depth(uav_desc.Texture3D.MipSlice);
        gpu_system.NativeDevice()->CreateUnorderedAccessView(texture.NativeTexture<D3D12Traits>(), nullptr, &uav_desc, cpu_handle_);
    }

    GpuUnorderedAccessView::GpuUnorderedAccessView(GpuSystem& gpu_system, GpuBuffer& buffer, GpuFormat format)
        : GpuUnorderedAccessView(gpu_system, buffer, 0, buffer.Size() / FormatSize(format), format)
    {
    }

    GpuUnorderedAccessView::GpuUnorderedAccessView(
        GpuSystem& gpu_system, GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, GpuFormat format)
        : gpu_system_(&gpu_system), buffer_(&buffer)
    {
        desc_block_ = gpu_system.AllocCbvSrvUavDescBlock(1);
        cpu_handle_ = desc_block_.CpuHandle();

        D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc{};
        uav_desc.Format = ToDxgiFormat(format);
        uav_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uav_desc.Buffer.FirstElement = first_element;
        uav_desc.Buffer.NumElements = num_elements;
        uav_desc.Buffer.StructureByteStride = 0;
        gpu_system.NativeDevice()->CreateUnorderedAccessView(buffer.NativeBuffer<D3D12Traits>(), nullptr, &uav_desc, cpu_handle_);
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
        gpu_system.NativeDevice()->CreateUnorderedAccessView(buffer.NativeBuffer<D3D12Traits>(), nullptr, &uav_desc, cpu_handle_);
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
        if (texture_2d_ != nullptr)
        {
            texture_2d_->Transition(cmd_list, GpuResourceState::UnorderedAccess);
        }
        else if (texture_2d_array_ != nullptr)
        {
            texture_2d_array_->Transition(cmd_list, GpuResourceState::UnorderedAccess);
        }
        else if (texture_3d_ != nullptr)
        {
            texture_3d_->Transition(cmd_list, GpuResourceState::UnorderedAccess);
        }
        else if (buffer_ != nullptr)
        {
            buffer_->Transition(cmd_list, GpuResourceState::UnorderedAccess);
        }
    }

    void GpuUnorderedAccessView::CopyTo(D3D12_CPU_DESCRIPTOR_HANDLE dst_handle) const noexcept
    {
        gpu_system_->NativeDevice()->CopyDescriptorsSimple(1, dst_handle, cpu_handle_, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }

    D3D12_CPU_DESCRIPTOR_HANDLE GpuUnorderedAccessView::CpuHandle() const noexcept
    {
        return cpu_handle_;
    }
} // namespace AIHoloImager
