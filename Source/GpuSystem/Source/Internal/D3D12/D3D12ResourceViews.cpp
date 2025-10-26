// Copyright (c) 2025 Minmin Gong
//

#include "D3D12ResourceViews.hpp"

#include "Base/MiniWindows.hpp"

#include <directx/d3d12.h>

#include "D3D12/D3D12Conversion.hpp"
#include "D3D12Buffer.hpp"
#include "D3D12System.hpp"
#include "D3D12Texture.hpp"

namespace AIHoloImager
{
    D3D12_IMP_IMP(ShaderResourceView)

    D3D12ShaderResourceView::D3D12ShaderResourceView(
        GpuSystem& gpu_system, const GpuTexture2D& texture, uint32_t sub_resource, GpuFormat format)
        : gpu_system_(&gpu_system), resource_(&texture)
    {
        desc_block_ = gpu_system.AllocCbvSrvUavDescBlock(1);
        cpu_handle_ = desc_block_.CpuHandle();

        auto* d3d12_device = D3D12Imp(*gpu_system_).Device();
        auto* d3d12_texture = D3D12Imp(texture).Resource();

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
        d3d12_device->CreateShaderResourceView(d3d12_texture, &srv_desc, ToD3D12CpuDescriptorHandle(cpu_handle_));
    }

    D3D12ShaderResourceView::D3D12ShaderResourceView(
        GpuSystem& gpu_system, const GpuTexture2DArray& texture_array, uint32_t sub_resource, GpuFormat format)
        : gpu_system_(&gpu_system), resource_(&texture_array)
    {
        desc_block_ = gpu_system.AllocCbvSrvUavDescBlock(1);
        cpu_handle_ = desc_block_.CpuHandle();

        auto* d3d12_device = D3D12Imp(*gpu_system_).Device();
        auto* d3d12_texture = D3D12Imp(texture_array).Resource();

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
        d3d12_device->CreateShaderResourceView(d3d12_texture, &srv_desc, ToD3D12CpuDescriptorHandle(cpu_handle_));
    }

    D3D12ShaderResourceView::D3D12ShaderResourceView(
        GpuSystem& gpu_system, const GpuTexture3D& texture, uint32_t sub_resource, GpuFormat format)
        : gpu_system_(&gpu_system), resource_(&texture)
    {
        desc_block_ = gpu_system.AllocCbvSrvUavDescBlock(1);
        cpu_handle_ = desc_block_.CpuHandle();

        auto* d3d12_device = D3D12Imp(*gpu_system_).Device();
        auto* d3d12_texture = D3D12Imp(texture).Resource();

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
        d3d12_device->CreateShaderResourceView(d3d12_texture, &srv_desc, ToD3D12CpuDescriptorHandle(cpu_handle_));
    }

    D3D12ShaderResourceView::D3D12ShaderResourceView(
        GpuSystem& gpu_system, const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, GpuFormat format)
        : gpu_system_(&gpu_system), resource_(&buffer)
    {
        desc_block_ = gpu_system.AllocCbvSrvUavDescBlock(1);
        cpu_handle_ = desc_block_.CpuHandle();

        auto* d3d12_device = D3D12Imp(*gpu_system_).Device();
        auto* d3d12_buff = D3D12Imp(buffer).Resource();

        D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc{};
        srv_desc.Format = ToDxgiFormat(format);
        srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv_desc.Buffer.FirstElement = first_element;
        srv_desc.Buffer.NumElements = num_elements;
        srv_desc.Buffer.StructureByteStride = 0;
        srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
        d3d12_device->CreateShaderResourceView(d3d12_buff, &srv_desc, ToD3D12CpuDescriptorHandle(cpu_handle_));
    }

    D3D12ShaderResourceView::D3D12ShaderResourceView(
        GpuSystem& gpu_system, const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size)
        : gpu_system_(&gpu_system), resource_(&buffer)
    {
        desc_block_ = gpu_system.AllocCbvSrvUavDescBlock(1);
        cpu_handle_ = desc_block_.CpuHandle();

        auto* d3d12_device = D3D12Imp(*gpu_system_).Device();
        auto* d3d12_buff = D3D12Imp(buffer).Resource();

        D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc{};
        srv_desc.Format = DXGI_FORMAT_UNKNOWN;
        srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv_desc.Buffer.FirstElement = first_element;
        srv_desc.Buffer.NumElements = num_elements;
        srv_desc.Buffer.StructureByteStride = element_size;
        srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
        d3d12_device->CreateShaderResourceView(d3d12_buff, &srv_desc, ToD3D12CpuDescriptorHandle(cpu_handle_));
    }

    D3D12ShaderResourceView::~D3D12ShaderResourceView()
    {
        this->Reset();
    }

    D3D12ShaderResourceView::D3D12ShaderResourceView(D3D12ShaderResourceView&& other) noexcept = default;
    D3D12ShaderResourceView::D3D12ShaderResourceView(GpuShaderResourceViewInternal&& other) noexcept
        : D3D12ShaderResourceView(static_cast<D3D12ShaderResourceView&&>(other))
    {
    }

    D3D12ShaderResourceView& D3D12ShaderResourceView::operator=(D3D12ShaderResourceView&& other) noexcept = default;
    GpuShaderResourceViewInternal& D3D12ShaderResourceView::operator=(GpuShaderResourceViewInternal&& other) noexcept
    {
        return this->operator=(static_cast<D3D12ShaderResourceView&&>(other));
    }

    void D3D12ShaderResourceView::Reset()
    {
        cpu_handle_ = {};
        if (desc_block_)
        {
            gpu_system_->DeallocCbvSrvUavDescBlock(std::move(desc_block_));
        }
    }

    void D3D12ShaderResourceView::Transition(GpuCommandList& cmd_list) const
    {
        resource_->Transition(cmd_list, GpuResourceState::Common);
    }

    void D3D12ShaderResourceView::Transition(D3D12CommandList& cmd_list) const
    {
        D3D12Imp(*resource_).Transition(cmd_list, GpuResourceState::Common);
    }

    void D3D12ShaderResourceView::CopyTo(GpuDescriptorCpuHandle dst_handle) const noexcept
    {
        auto* d3d12_device = D3D12Imp(*gpu_system_).Device();
        d3d12_device->CopyDescriptorsSimple(
            1, ToD3D12CpuDescriptorHandle(dst_handle), ToD3D12CpuDescriptorHandle(cpu_handle_), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }

    GpuDescriptorCpuHandle D3D12ShaderResourceView::CpuHandle() const noexcept
    {
        return cpu_handle_;
    }


    D3D12_IMP_IMP(RenderTargetView)

    D3D12RenderTargetView::D3D12RenderTargetView(GpuSystem& gpu_system, GpuTexture2D& texture, GpuFormat format)
        : gpu_system_(&gpu_system), resource_(&texture)
    {
        desc_block_ = gpu_system.AllocRtvDescBlock(1);
        cpu_handle_ = desc_block_.CpuHandle();

        auto* d3d12_device = D3D12Imp(*gpu_system_).Device();
        auto* d3d12_texture = D3D12Imp(texture).Resource();

        D3D12_RENDER_TARGET_VIEW_DESC rtv_desc{};
        rtv_desc.Format = ToDxgiFormat(format == GpuFormat::Unknown ? texture.Format() : format);
        rtv_desc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        d3d12_device->CreateRenderTargetView(d3d12_texture, &rtv_desc, ToD3D12CpuDescriptorHandle(cpu_handle_));
    }

    D3D12RenderTargetView::~D3D12RenderTargetView()
    {
        this->Reset();
    }

    D3D12RenderTargetView::D3D12RenderTargetView(D3D12RenderTargetView&& other) noexcept = default;
    D3D12RenderTargetView::D3D12RenderTargetView(GpuRenderTargetViewInternal&& other) noexcept
        : D3D12RenderTargetView(static_cast<D3D12RenderTargetView&&>(other))
    {
    }
    D3D12RenderTargetView& D3D12RenderTargetView::operator=(D3D12RenderTargetView&& other) noexcept = default;
    GpuRenderTargetViewInternal& D3D12RenderTargetView::operator=(GpuRenderTargetViewInternal&& other) noexcept
    {
        return this->operator=(static_cast<D3D12RenderTargetView&&>(other));
    }

    D3D12RenderTargetView::operator bool() const noexcept
    {
        return (cpu_handle_.handle != 0);
    }

    void D3D12RenderTargetView::Reset()
    {
        cpu_handle_ = {};
        if (desc_block_)
        {
            gpu_system_->DeallocRtvDescBlock(std::move(desc_block_));
        }
    }

    void D3D12RenderTargetView::Transition(GpuCommandList& cmd_list) const
    {
        resource_->Transition(cmd_list, GpuResourceState::ColorWrite);
    }

    void D3D12RenderTargetView::Transition(D3D12CommandList& cmd_list) const
    {
        D3D12Imp(*resource_).Transition(cmd_list, GpuResourceState::ColorWrite);
    }

    void D3D12RenderTargetView::CopyTo(GpuDescriptorCpuHandle dst_handle) const noexcept
    {
        auto* d3d12_device = D3D12Imp(*gpu_system_).Device();
        d3d12_device->CopyDescriptorsSimple(
            1, ToD3D12CpuDescriptorHandle(dst_handle), ToD3D12CpuDescriptorHandle(cpu_handle_), D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    }

    GpuDescriptorCpuHandle D3D12RenderTargetView::CpuHandle() const noexcept
    {
        return cpu_handle_;
    }


    D3D12_IMP_IMP(DepthStencilView)

    D3D12DepthStencilView::D3D12DepthStencilView(GpuSystem& gpu_system, GpuTexture2D& texture, GpuFormat format)
        : gpu_system_(&gpu_system), resource_(&texture)
    {
        desc_block_ = gpu_system.AllocDsvDescBlock(1);
        cpu_handle_ = desc_block_.CpuHandle();

        auto* d3d12_device = D3D12Imp(*gpu_system_).Device();
        auto* d3d12_texture = D3D12Imp(texture).Resource();

        D3D12_DEPTH_STENCIL_VIEW_DESC dsv_desc{};
        dsv_desc.Format = ToDxgiFormat(format == GpuFormat::Unknown ? texture.Format() : format);
        dsv_desc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
        d3d12_device->CreateDepthStencilView(d3d12_texture, &dsv_desc, ToD3D12CpuDescriptorHandle(cpu_handle_));
    }

    D3D12DepthStencilView::~D3D12DepthStencilView()
    {
        this->Reset();
    }

    D3D12DepthStencilView::D3D12DepthStencilView(D3D12DepthStencilView&& other) noexcept = default;
    D3D12DepthStencilView::D3D12DepthStencilView(GpuDepthStencilViewInternal&& other) noexcept
        : D3D12DepthStencilView(static_cast<D3D12DepthStencilView&&>(other))
    {
    }
    D3D12DepthStencilView& D3D12DepthStencilView::operator=(D3D12DepthStencilView&& other) noexcept = default;
    GpuDepthStencilViewInternal& D3D12DepthStencilView::operator=(GpuDepthStencilViewInternal&& other) noexcept
    {
        return this->operator=(static_cast<D3D12DepthStencilView&&>(other));
    }

    D3D12DepthStencilView::operator bool() const noexcept
    {
        return (cpu_handle_.handle != 0);
    }

    void D3D12DepthStencilView::Reset()
    {
        cpu_handle_ = {};
        if (desc_block_)
        {
            gpu_system_->DeallocDsvDescBlock(std::move(desc_block_));
        }
    }

    void D3D12DepthStencilView::Transition(GpuCommandList& cmd_list) const
    {
        resource_->Transition(cmd_list, GpuResourceState::DepthWrite);
    }

    void D3D12DepthStencilView::Transition(D3D12CommandList& cmd_list) const
    {
        D3D12Imp(*resource_).Transition(cmd_list, GpuResourceState::DepthWrite);
    }

    void D3D12DepthStencilView::CopyTo(GpuDescriptorCpuHandle dst_handle) const noexcept
    {
        auto* d3d12_device = D3D12Imp(*gpu_system_).Device();
        d3d12_device->CopyDescriptorsSimple(
            1, ToD3D12CpuDescriptorHandle(dst_handle), ToD3D12CpuDescriptorHandle(cpu_handle_), D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
    }

    GpuDescriptorCpuHandle D3D12DepthStencilView::CpuHandle() const noexcept
    {
        return cpu_handle_;
    }


    D3D12_IMP_IMP(UnorderedAccessView)

    D3D12UnorderedAccessView::D3D12UnorderedAccessView(
        GpuSystem& gpu_system, GpuTexture2D& texture, uint32_t sub_resource, GpuFormat format)
        : gpu_system_(&gpu_system), resource_(&texture)
    {
        desc_block_ = gpu_system.AllocCbvSrvUavDescBlock(1);
        cpu_handle_ = desc_block_.CpuHandle();

        auto* d3d12_device = D3D12Imp(*gpu_system_).Device();
        auto* d3d12_texture = D3D12Imp(texture).Resource();

        D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc{};
        uav_desc.Format = ToDxgiFormat(format == GpuFormat::Unknown ? texture.Format() : format);
        uav_desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        uint32_t array_slice;
        DecomposeSubResource(sub_resource, texture.MipLevels(), 1, uav_desc.Texture2D.MipSlice, array_slice, uav_desc.Texture2D.PlaneSlice);
        d3d12_device->CreateUnorderedAccessView(d3d12_texture, nullptr, &uav_desc, ToD3D12CpuDescriptorHandle(cpu_handle_));
    }

    D3D12UnorderedAccessView::D3D12UnorderedAccessView(
        GpuSystem& gpu_system, GpuTexture2DArray& texture_array, uint32_t sub_resource, GpuFormat format)
        : gpu_system_(&gpu_system), resource_(&texture_array)
    {
        desc_block_ = gpu_system.AllocCbvSrvUavDescBlock(1);
        cpu_handle_ = desc_block_.CpuHandle();

        auto* d3d12_device = D3D12Imp(*gpu_system_).Device();
        auto* d3d12_texture = D3D12Imp(texture_array).Resource();

        D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc{};
        uav_desc.Format = ToDxgiFormat(format == GpuFormat::Unknown ? texture_array.Format() : format);
        uav_desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        uav_desc.Texture2DArray.ArraySize = 1;
        DecomposeSubResource(sub_resource, texture_array.MipLevels(), texture_array.ArraySize(), uav_desc.Texture2DArray.MipSlice,
            uav_desc.Texture2DArray.FirstArraySlice, uav_desc.Texture2DArray.PlaneSlice);
        d3d12_device->CreateUnorderedAccessView(d3d12_texture, nullptr, &uav_desc, ToD3D12CpuDescriptorHandle(cpu_handle_));
    }

    D3D12UnorderedAccessView::D3D12UnorderedAccessView(
        GpuSystem& gpu_system, GpuTexture3D& texture, uint32_t sub_resource, GpuFormat format)
        : gpu_system_(&gpu_system), resource_(&texture)
    {
        desc_block_ = gpu_system.AllocCbvSrvUavDescBlock(1);
        cpu_handle_ = desc_block_.CpuHandle();

        auto* d3d12_device = D3D12Imp(*gpu_system_).Device();
        auto* d3d12_texture = D3D12Imp(texture).Resource();

        D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc{};
        uav_desc.Format = ToDxgiFormat(format == GpuFormat::Unknown ? texture.Format() : format);
        uav_desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
        uint32_t array_slice;
        uint32_t plane_slice;
        DecomposeSubResource(sub_resource, texture.MipLevels(), 1, uav_desc.Texture3D.MipSlice, array_slice, plane_slice);
        uav_desc.Texture3D.FirstWSlice = 0;
        uav_desc.Texture3D.WSize = texture.Depth(uav_desc.Texture3D.MipSlice);
        d3d12_device->CreateUnorderedAccessView(d3d12_texture, nullptr, &uav_desc, ToD3D12CpuDescriptorHandle(cpu_handle_));
    }

    D3D12UnorderedAccessView::D3D12UnorderedAccessView(
        GpuSystem& gpu_system, GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, GpuFormat format)
        : gpu_system_(&gpu_system), resource_(&buffer)
    {
        desc_block_ = gpu_system.AllocCbvSrvUavDescBlock(1);
        cpu_handle_ = desc_block_.CpuHandle();

        auto* d3d12_device = D3D12Imp(*gpu_system_).Device();
        auto* d3d12_buff = D3D12Imp(buffer).Resource();

        D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc{};
        uav_desc.Format = ToDxgiFormat(format);
        uav_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uav_desc.Buffer.FirstElement = first_element;
        uav_desc.Buffer.NumElements = num_elements;
        uav_desc.Buffer.StructureByteStride = 0;
        d3d12_device->CreateUnorderedAccessView(d3d12_buff, nullptr, &uav_desc, ToD3D12CpuDescriptorHandle(cpu_handle_));
    }

    D3D12UnorderedAccessView::D3D12UnorderedAccessView(
        GpuSystem& gpu_system, GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size)
        : gpu_system_(&gpu_system), resource_(&buffer)
    {
        desc_block_ = gpu_system.AllocCbvSrvUavDescBlock(1);
        cpu_handle_ = desc_block_.CpuHandle();

        auto* d3d12_device = D3D12Imp(*gpu_system_).Device();
        auto* d3d12_buff = D3D12Imp(buffer).Resource();

        D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc{};
        uav_desc.Format = DXGI_FORMAT_UNKNOWN;
        uav_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uav_desc.Buffer.FirstElement = first_element;
        uav_desc.Buffer.NumElements = num_elements;
        uav_desc.Buffer.StructureByteStride = element_size;
        d3d12_device->CreateUnorderedAccessView(d3d12_buff, nullptr, &uav_desc, ToD3D12CpuDescriptorHandle(cpu_handle_));
    }

    D3D12UnorderedAccessView::~D3D12UnorderedAccessView()
    {
        this->Reset();
    }

    D3D12UnorderedAccessView::D3D12UnorderedAccessView(D3D12UnorderedAccessView&& other) noexcept = default;
    D3D12UnorderedAccessView::D3D12UnorderedAccessView(GpuUnorderedAccessViewInternal&& other) noexcept
        : D3D12UnorderedAccessView(static_cast<D3D12UnorderedAccessView&&>(other))
    {
    }
    D3D12UnorderedAccessView& D3D12UnorderedAccessView::operator=(D3D12UnorderedAccessView&& other) noexcept = default;
    GpuUnorderedAccessViewInternal& D3D12UnorderedAccessView::operator=(GpuUnorderedAccessViewInternal&& other) noexcept
    {
        return this->operator=(static_cast<D3D12UnorderedAccessView&&>(other));
    }

    void D3D12UnorderedAccessView::Reset()
    {
        cpu_handle_ = {};
        if (desc_block_)
        {
            gpu_system_->DeallocCbvSrvUavDescBlock(std::move(desc_block_));
        }
    }

    void D3D12UnorderedAccessView::Transition(GpuCommandList& cmd_list) const
    {
        resource_->Transition(cmd_list, GpuResourceState::UnorderedAccess);
    }

    void D3D12UnorderedAccessView::Transition(D3D12CommandList& cmd_list) const
    {
        D3D12Imp(*resource_).Transition(cmd_list, GpuResourceState::UnorderedAccess);
    }

    void D3D12UnorderedAccessView::CopyTo(GpuDescriptorCpuHandle dst_handle) const noexcept
    {
        auto* d3d12_device = D3D12Imp(*gpu_system_).Device();
        d3d12_device->CopyDescriptorsSimple(
            1, ToD3D12CpuDescriptorHandle(dst_handle), ToD3D12CpuDescriptorHandle(cpu_handle_), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }

    GpuDescriptorCpuHandle D3D12UnorderedAccessView::CpuHandle() const noexcept
    {
        return cpu_handle_;
    }

    GpuResource* D3D12UnorderedAccessView::Resource() noexcept
    {
        return resource_;
    }
} // namespace AIHoloImager
