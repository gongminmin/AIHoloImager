// Copyright (c) 2025 Minmin Gong
//

#include "GpuDiffRenderTorch.hpp"

#include <glm/vec4.hpp>

#include "Base/ErrorHandling.hpp"

using namespace torch::autograd;

namespace AIHoloImager
{
    GpuDiffRenderTorch::GpuDiffRenderTorch(size_t gpu_system, torch::Device torch_device)
        : gpu_system_(*reinterpret_cast<GpuSystem*>(gpu_system)), gpu_dr_(gpu_system_), torch_device_(torch_device)
    {
        uses_cuda_copy_ = false;
        if (torch_device.is_cuda() && cuda_rt_)
        {
            int device_index = 0;
            if (torch_device.has_index())
            {
                device_index = torch_device.index();
            }

            MiniCudaRt::DeviceProp device_prop{};
            cuda_rt_.GetDeviceProperties(&device_prop, device_index);

            const LUID gpu_luid = gpu_system_.NativeDevice()->GetAdapterLuid();

            uses_cuda_copy_ = (std::memcmp(&gpu_luid, device_prop.luid, sizeof(gpu_luid)) == 0);
        }

        if (uses_cuda_copy_)
        {
            MiniCudaRt::ExternalSemaphoreHandleDesc ext_semaphore_handle_desc{};
            ext_semaphore_handle_desc.type = MiniCudaRt::ExternalSemaphoreHandleType::D3D12Fence;
            ext_semaphore_handle_desc.handle.win32.handle = gpu_system_.SharedFenceHandle();
            ext_semaphore_handle_desc.flags = 0;
            cuda_rt_.ImportExternalSemaphore(&ext_semaphore_, &ext_semaphore_handle_desc);

            cuda_rt_.StreamCreate(&copy_stream_);
        }
    }

    GpuDiffRenderTorch::~GpuDiffRenderTorch()
    {
        if (uses_cuda_copy_)
        {
            cuda_rt_.DestroyExternalSemaphore(ext_semaphore_);
            cuda_rt_.StreamDestroy(copy_stream_);
        }
    }

    tensor_list GpuDiffRenderTorch::Rasterize(torch::Tensor positions, torch::Tensor indices, std::tuple<uint32_t, uint32_t> resolution)
    {
        struct RasterizeAutogradFunc : public Function<RasterizeAutogradFunc>
        {
            static tensor_list forward(AutogradContext* ctx, GpuDiffRenderTorch* dr, torch::Tensor positions, torch::Tensor indices,
                std::tuple<uint32_t, uint32_t> resolution)
            {
                auto [barycentric, prim_id] = dr->RasterizeFwd(std::move(positions), std::move(indices), resolution);
                ctx->saved_data["dr"] = reinterpret_cast<int64_t>(dr);
                return {std::move(barycentric), std::move(prim_id)};
            }

            static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs)
            {
                auto* dr = reinterpret_cast<GpuDiffRenderTorch*>(ctx->saved_data["dr"].to<int64_t>());

                torch::Tensor grad_barycentric = std::move(grad_outputs[0]);
                auto grad_positions = dr->RasterizeBwd(std::move(grad_barycentric));
                return {torch::Tensor(), std::move(grad_positions), torch::Tensor(), torch::Tensor()};
            }
        };

        return RasterizeAutogradFunc::apply(this, std::move(positions), std::move(indices), resolution);
    }

    std::tuple<torch::Tensor, torch::Tensor> GpuDiffRenderTorch::RasterizeFwd(
        torch::Tensor positions, torch::Tensor indices, std::tuple<uint32_t, uint32_t> resolution)
    {
        const uint32_t width = std::get<1>(resolution);
        const uint32_t height = std::get<0>(resolution);

        auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

        this->Convert(cmd_list, std::move(positions), rast_intermediate_.positions, GpuHeap::Default, GpuResourceFlag::None,
            L"GpuDiffRenderTorch.RasterizeFwd.positions");
        this->Convert(cmd_list, std::move(indices), rast_intermediate_.indices, GpuHeap::Default, GpuResourceFlag::None,
            L"GpuDiffRenderTorch.RasterizeFwd.indices");

        gpu_dr_.RasterizeFwd(cmd_list, rast_intermediate_.positions, rast_intermediate_.indices, width, height,
            rast_intermediate_.barycentric, rast_intermediate_.prim_id);

        torch::Tensor barycentric = this->Convert(cmd_list, rast_intermediate_.barycentric);
        torch::Tensor prim_id = this->Convert(cmd_list, rast_intermediate_.prim_id);

        gpu_system_.Execute(std::move(cmd_list));

        return {std::move(barycentric), std::move(prim_id)};
    }

    torch::Tensor GpuDiffRenderTorch::RasterizeBwd(torch::Tensor grad_barycentric)
    {
        const uint32_t num_vertices = rast_intermediate_.positions.Size() / sizeof(glm::vec4);

        auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

        this->Convert(cmd_list, std::move(grad_barycentric), rast_intermediate_.grad_barycentric, GpuFormat::RG32_Float,
            GpuResourceFlag::None, L"GpuDiffRenderTorch.RasterizeBwd.grad_barycentric");

        gpu_dr_.RasterizeBwd(cmd_list, rast_intermediate_.positions, rast_intermediate_.indices, rast_intermediate_.barycentric,
            rast_intermediate_.prim_id, rast_intermediate_.grad_barycentric, rast_intermediate_.grad_positions);

        const torch::Tensor grad_positions = this->Convert(cmd_list, rast_intermediate_.grad_positions, {num_vertices, 4}, torch::kFloat32);

        gpu_system_.Execute(std::move(cmd_list));

        return grad_positions;
    }

    torch::Tensor GpuDiffRenderTorch::Interpolate(
        torch::Tensor vtx_attribs, torch::Tensor barycentric, torch::Tensor prim_id, torch::Tensor indices)
    {
        struct InterpolateAutogradFunc : public Function<InterpolateAutogradFunc>
        {
            static torch::Tensor forward(AutogradContext* ctx, GpuDiffRenderTorch* dr, torch::Tensor vtx_attribs, torch::Tensor barycentric,
                torch::Tensor prim_id, torch::Tensor indices)
            {
                auto shading = dr->InterpolateFwd(std::move(vtx_attribs), std::move(barycentric), std::move(prim_id), std::move(indices));
                ctx->saved_data["dr"] = reinterpret_cast<int64_t>(dr);
                return shading;
            }

            static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs)
            {
                auto* dr = reinterpret_cast<GpuDiffRenderTorch*>(ctx->saved_data["dr"].to<int64_t>());

                torch::Tensor grad_shading = std::move(grad_outputs[0]);
                auto [grad_vtx_attribs, grad_barycentric] = dr->InterpolateBwd(std::move(grad_shading));
                return {torch::Tensor(), std::move(grad_vtx_attribs), std::move(grad_barycentric), torch::Tensor(), torch::Tensor()};
            }
        };

        return InterpolateAutogradFunc::apply(this, std::move(vtx_attribs), std::move(barycentric), std::move(prim_id), std::move(indices));
    }

    torch::Tensor GpuDiffRenderTorch::InterpolateFwd(
        torch::Tensor vtx_attribs, torch::Tensor barycentric, torch::Tensor prim_id, torch::Tensor indices)
    {
        const uint32_t mini_batch = static_cast<uint32_t>(barycentric.size(0));
        const uint32_t width = static_cast<uint32_t>(barycentric.size(2));
        const uint32_t height = static_cast<uint32_t>(barycentric.size(1));
        const uint32_t num_attribs = static_cast<uint32_t>(vtx_attribs.size(1));

        auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

        interpolate_intermediate_.num_attribs = num_attribs;
        this->Convert(cmd_list, std::move(vtx_attribs), interpolate_intermediate_.vtx_attribs, GpuHeap::Default, GpuResourceFlag::None,
            L"GpuDiffRenderTorch.InterpolateFwd.vtx_attribs");
        this->Convert(cmd_list, std::move(barycentric), interpolate_intermediate_.barycentric, GpuFormat::RG32_Float, GpuResourceFlag::None,
            L"GpuDiffRenderTorch.InterpolateFwd.barycentric");
        this->Convert(cmd_list, std::move(prim_id), interpolate_intermediate_.prim_id, GpuFormat::R32_Uint, GpuResourceFlag::None,
            L"GpuDiffRenderTorch.InterpolateFwd.prim_id");
        this->Convert(cmd_list, std::move(indices), interpolate_intermediate_.indices, GpuHeap::Default, GpuResourceFlag::None,
            L"GpuDiffRenderTorch.InterpolateFwd.indices");

        gpu_dr_.InterpolateFwd(cmd_list, interpolate_intermediate_.vtx_attribs, num_attribs, interpolate_intermediate_.barycentric,
            interpolate_intermediate_.prim_id, interpolate_intermediate_.indices, interpolate_intermediate_.shading);

        const torch::Tensor shading =
            this->Convert(cmd_list, interpolate_intermediate_.shading, {mini_batch, height, width, num_attribs}, torch::kFloat32);

        gpu_system_.Execute(std::move(cmd_list));

        return shading;
    }

    std::tuple<torch::Tensor, torch::Tensor> GpuDiffRenderTorch::InterpolateBwd(torch::Tensor grad_shading)
    {
        const uint32_t num_attribs = interpolate_intermediate_.num_attribs;
        const uint32_t num_vertices = interpolate_intermediate_.vtx_attribs.Size() / (num_attribs * sizeof(float));

        auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

        this->Convert(cmd_list, std::move(grad_shading), interpolate_intermediate_.grad_shading, GpuHeap::Default, GpuResourceFlag::None,
            L"GpuDiffRenderTorch.InterpolateBwd.grad_shading");

        gpu_dr_.InterpolateBwd(cmd_list, interpolate_intermediate_.vtx_attribs, num_attribs, interpolate_intermediate_.barycentric,
            interpolate_intermediate_.prim_id, interpolate_intermediate_.indices, interpolate_intermediate_.grad_shading,
            interpolate_intermediate_.grad_vtx_attribs, interpolate_intermediate_.grad_barycentric);

        torch::Tensor grad_vtx_attribs =
            this->Convert(cmd_list, interpolate_intermediate_.grad_vtx_attribs, {num_vertices, num_attribs}, torch::kFloat32);
        torch::Tensor grad_barycentric = this->Convert(cmd_list, interpolate_intermediate_.grad_barycentric);

        gpu_system_.Execute(std::move(cmd_list));

        return {std::move(grad_vtx_attribs), std::move(grad_barycentric)};
    }

    GpuDiffRenderTorch::AntiAliasOppositeVertices GpuDiffRenderTorch::AntiAliasConstructOppositeVertices(torch::Tensor indices)
    {
        auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

        GpuBuffer indices_buff;
        this->Convert(cmd_list, std::move(indices), indices_buff, GpuHeap::Default, GpuResourceFlag::None,
            L"GpuDiffRenderTorch.AntiAliasConstructOppositeVertices.indices");

        AntiAliasOppositeVertices ret;
        gpu_dr_.AntiAliasConstructOppositeVertices(cmd_list, indices_buff, ret.opposite_vertices);

        gpu_system_.Execute(std::move(cmd_list));

        return ret;
    }

    torch::Tensor GpuDiffRenderTorch::AntiAlias(torch::Tensor shading, torch::Tensor prim_id, torch::Tensor positions,
        torch::Tensor indices, const AntiAliasOppositeVertices* opposite_vertices)
    {
        struct AntiAliasAutogradFunc : public Function<AntiAliasAutogradFunc>
        {
            static torch::Tensor forward(AutogradContext* ctx, GpuDiffRenderTorch* dr, torch::Tensor shading, torch::Tensor prim_id,
                torch::Tensor positions, torch::Tensor indices, const AntiAliasOppositeVertices* opposite_vertices)
            {
                auto anti_aliased =
                    dr->AntiAliasFwd(std::move(shading), std::move(prim_id), std::move(positions), std::move(indices), opposite_vertices);
                ctx->saved_data["dr"] = reinterpret_cast<int64_t>(dr);
                return anti_aliased;
            }

            static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs)
            {
                auto* dr = reinterpret_cast<GpuDiffRenderTorch*>(ctx->saved_data["dr"].to<int64_t>());

                torch::Tensor grad_anti_aliased = std::move(grad_outputs[0]);
                auto [grad_shading, grad_positions] = dr->AntiAliasBwd(std::move(grad_anti_aliased));
                return {
                    torch::Tensor(), std::move(grad_shading), torch::Tensor(), std::move(grad_positions), torch::Tensor(), torch::Tensor()};
            }
        };

        return AntiAliasAutogradFunc::apply(
            this, std::move(shading), std::move(prim_id), std::move(positions), std::move(indices), opposite_vertices);
    }

    torch::Tensor GpuDiffRenderTorch::AntiAliasFwd(torch::Tensor shading, torch::Tensor prim_id, torch::Tensor positions,
        torch::Tensor indices, const AntiAliasOppositeVertices* opposite_vertices)
    {
        AntiAliasOppositeVertices new_opposite_vertices;
        if (opposite_vertices == nullptr)
        {
            new_opposite_vertices = this->AntiAliasConstructOppositeVertices(indices);
            opposite_vertices = &new_opposite_vertices;
        }

        const uint32_t width = static_cast<uint32_t>(shading.size(2));
        const uint32_t height = static_cast<uint32_t>(shading.size(1));
        const uint32_t num_attribs = static_cast<uint32_t>(shading.size(3));

        auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

        this->Convert(cmd_list, std::move(shading), aa_intermediate_.shading, GpuHeap::Default, GpuResourceFlag::None,
            L"GpuDiffRenderTorch.AntiAliasFwd.shading");
        this->Convert(cmd_list, std::move(prim_id), aa_intermediate_.prim_id, GpuFormat::R32_Uint, GpuResourceFlag::None,
            L"GpuDiffRenderTorch.AntiAliasFwd.prim_id");
        this->Convert(cmd_list, std::move(positions), aa_intermediate_.positions, GpuHeap::Default, GpuResourceFlag::None,
            L"GpuDiffRenderTorch.AntiAliasFwd.positions");
        this->Convert(cmd_list, std::move(indices), aa_intermediate_.indices, GpuHeap::Default, GpuResourceFlag::None,
            L"GpuDiffRenderTorch.AntiAliasFwd.indices");

        gpu_dr_.AntiAliasFwd(cmd_list, aa_intermediate_.shading, aa_intermediate_.prim_id, aa_intermediate_.positions,
            aa_intermediate_.indices, opposite_vertices->opposite_vertices, aa_intermediate_.anti_aliased);

        torch::Tensor anti_aliased =
            this->Convert(cmd_list, aa_intermediate_.anti_aliased, {1, height, width, num_attribs}, torch::kFloat32);

        gpu_system_.Execute(std::move(cmd_list));

        return anti_aliased;
    }

    std::tuple<torch::Tensor, torch::Tensor> GpuDiffRenderTorch::AntiAliasBwd(torch::Tensor grad_anti_aliased)
    {
        const uint32_t num_vertices = aa_intermediate_.positions.Size() / sizeof(glm::vec4);
        const uint32_t mini_batch = 1;
        const uint32_t width = aa_intermediate_.prim_id.Width(0);
        const uint32_t height = aa_intermediate_.prim_id.Height(0);
        const uint32_t num_attribs = aa_intermediate_.shading.Size() / (width * height * sizeof(float));

        auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

        this->Convert(cmd_list, std::move(grad_anti_aliased), aa_intermediate_.grad_anti_aliased, GpuHeap::Default, GpuResourceFlag::None,
            L"GpuDiffRenderTorch.AntiAliasBwd.grad_anti_aliased");

        gpu_dr_.AntiAliasBwd(cmd_list, aa_intermediate_.shading, aa_intermediate_.prim_id, aa_intermediate_.positions,
            aa_intermediate_.indices, aa_intermediate_.grad_anti_aliased, aa_intermediate_.grad_shading, aa_intermediate_.grad_positions);

        torch::Tensor grad_shading =
            this->Convert(cmd_list, aa_intermediate_.grad_shading, {mini_batch, height, width, num_attribs}, torch::kFloat32);
        torch::Tensor grad_positions = this->Convert(cmd_list, aa_intermediate_.grad_positions, {num_vertices, 4}, torch::kFloat32);

        gpu_system_.Execute(std::move(cmd_list));

        return {std::move(grad_shading), std::move(grad_positions)};
    }

    void GpuDiffRenderTorch::Convert(
        GpuCommandList& cmd_list, torch::Tensor tensor, GpuBuffer& buff, GpuHeap heap, GpuResourceFlag flags, std::wstring_view name)
    {
        assert(tensor.device() == torch_device_);

        uint32_t size = static_cast<uint32_t>(tensor.element_size());
        for (uint32_t i = 0; i < tensor.sizes().size(); ++i)
        {
            size *= static_cast<uint32_t>(tensor.size(i));
        }

        if (uses_cuda_copy_)
        {
            flags |= GpuResourceFlag::Shareable;
        }

        if (buff.Size() != size)
        {
            buff = GpuBuffer(gpu_system_, size, heap, flags);
        }
        buff.Name(std::move(name));

        if (uses_cuda_copy_)
        {
            GpuBuffer default_buff;
            GpuBuffer* copy_buff;
            if (heap != GpuHeap::Default)
            {
                default_buff = GpuBuffer(gpu_system_, size, GpuHeap::Default, flags);
                copy_buff = &default_buff;
            }
            else
            {
                copy_buff = &buff;
            }

            MiniCudaRt::ExternalMemory_t ext_mem = this->ImportFromResource(buff);

            MiniCudaRt::ExternalMemoryBufferDesc ext_mem_buffer_desc{};
            ext_mem_buffer_desc.size = size;
            ext_mem_buffer_desc.offset = 0;
            ext_mem_buffer_desc.flags = 0;

            void* ext_mem_ptr;
            cuda_rt_.ExternalMemoryGetMappedBuffer(&ext_mem_ptr, ext_mem, &ext_mem_buffer_desc);

            cuda_rt_.MemcpyAsync(ext_mem_ptr, tensor.const_data_ptr(), size, MiniCudaRt::MemcpyKind::DeviceToDevice, copy_stream_);

            cuda_rt_.Free(ext_mem_ptr);
            cuda_rt_.DestroyExternalMemory(ext_mem);

            const uint64_t fence_val = gpu_system_.FenceValue() + 1;
            this->SignalExternalSemaphore(fence_val);
            gpu_system_.GpuWait(GpuSystem::CmdQueueType::Render, fence_val);

            if (heap != GpuHeap::Default)
            {
                cmd_list.Copy(buff, *copy_buff);
            }
        }
        else
        {
            tensor = tensor.cpu();
            if (heap == GpuHeap::Default)
            {
                const GpuUploadBuffer upload_buff(gpu_system_, tensor.const_data_ptr(), size);
                cmd_list.Copy(buff, upload_buff);
            }
            else
            {
                std::memcpy(buff.Map(), tensor.const_data_ptr(), buff.Size());
                buff.Unmap();
            }
        }
    }

    void GpuDiffRenderTorch::Convert(
        GpuCommandList& cmd_list, torch::Tensor tensor, GpuTexture2D& tex, GpuFormat format, GpuResourceFlag flags, std::wstring_view name)
    {
        assert(tensor.device() == torch_device_);
        assert(tensor.size(-1) == FormatChannels(format));
        assert(tensor.element_size() == FormatChannelSize(format));

        if (uses_cuda_copy_)
        {
            flags |= GpuResourceFlag::Shareable;
        }

        const uint32_t width = static_cast<uint32_t>(tensor.size(-2));
        const uint32_t height = static_cast<uint32_t>(tensor.size(-3));

        if ((tex.Width(0) != width) || (tex.Height(0) != height) || (tex.Format() != format))
        {
            tex = GpuTexture2D(gpu_system_, width, height, 1, format, flags);
        }
        tex.Name(std::move(name));

        if (uses_cuda_copy_)
        {
            MiniCudaRt::ExternalMemory_t ext_mem = this->ImportFromResource(tex);

            MiniCudaRt::ExternalMemoryMipmappedArrayDesc ext_mem_mip_desc{};
            ext_mem_mip_desc.extent = {width, height, 1};
            ext_mem_mip_desc.format_desc = this->FormatDesc(format);
            ext_mem_mip_desc.num_levels = 1;
            ext_mem_mip_desc.flags = MiniCudaRt::ArraySurfaceLoadStore;

            MiniCudaRt::MipmappedArray_t cu_mip_array;
            cuda_rt_.ExternalMemoryGetMappedMipmappedArray(&cu_mip_array, ext_mem, &ext_mem_mip_desc);

            MiniCudaRt::Array_t cu_array;
            cuda_rt_.GetMipmappedArrayLevel(&cu_array, cu_mip_array, 0);

            tensor = tensor.contiguous();

            MiniCudaRt::Memcpy3DParams p{};
            p.src_ptr.ptr = tensor.mutable_data_ptr();
            p.src_ptr.pitch = width * FormatSize(format);
            p.src_ptr.x_size = width;
            p.src_ptr.y_size = height;
            p.dst_array = cu_array;
            p.extent = ext_mem_mip_desc.extent;
            p.kind = MiniCudaRt::MemcpyKind::DeviceToDevice;
            cuda_rt_.Memcpy3DAsync(&p, copy_stream_);

            cuda_rt_.FreeMipmappedArray(cu_mip_array);
            cuda_rt_.DestroyExternalMemory(ext_mem);

            const uint64_t fence_val = gpu_system_.FenceValue() + 1;
            this->SignalExternalSemaphore(fence_val);
            gpu_system_.GpuWait(GpuSystem::CmdQueueType::Render, fence_val);
        }
        else
        {
            tensor = tensor.cpu();
            tex.Upload(gpu_system_, cmd_list, 0, tensor.const_data_ptr());
        }
    }

    torch::Tensor GpuDiffRenderTorch::Convert(
        GpuCommandList& cmd_list, const GpuBuffer& buff, const torch::IntArrayRef& size, torch::Dtype data_type)
    {
        auto opts = torch::TensorOptions().dtype(data_type);
        torch::Tensor tensor;
        if (uses_cuda_copy_)
        {
            uint64_t fence_val = gpu_system_.ExecuteAndReset(cmd_list);
            this->WaitExternalSemaphore(fence_val);

            MiniCudaRt::ExternalMemory_t ext_mem = this->ImportFromResource(buff);

            MiniCudaRt::ExternalMemoryBufferDesc ext_mem_buffer_desc{};
            ext_mem_buffer_desc.size = buff.Size();
            ext_mem_buffer_desc.offset = 0;
            ext_mem_buffer_desc.flags = 0;

            void* ext_mem_ptr;
            cuda_rt_.ExternalMemoryGetMappedBuffer(&ext_mem_ptr, ext_mem, &ext_mem_buffer_desc);

            opts = opts.device(torch_device_);
            tensor = torch::empty(size, opts);

            cuda_rt_.MemcpyAsync(tensor.mutable_data_ptr(), ext_mem_ptr, buff.Size(), MiniCudaRt::MemcpyKind::DeviceToDevice, copy_stream_);

            cuda_rt_.Free(ext_mem_ptr);
            cuda_rt_.DestroyExternalMemory(ext_mem);

            ++fence_val;
            this->SignalExternalSemaphore(fence_val);
            gpu_system_.GpuWait(GpuSystem::CmdQueueType::Render, fence_val);
        }
        else
        {
            GpuReadbackBuffer read_back_buff(gpu_system_, buff.Size());
            cmd_list.Copy(read_back_buff, buff);

            gpu_system_.ExecuteAndReset(cmd_list);
            gpu_system_.CpuWait();

            opts = opts.device(torch::kCPU);
            tensor = torch::from_blob(read_back_buff.MappedData(), size, opts);
            tensor = tensor.to(torch_device_);
        }

        return tensor;
    }

    torch::Tensor GpuDiffRenderTorch::Convert(GpuCommandList& cmd_list, const GpuTexture2D& tex)
    {
        const uint32_t width = tex.Width(0);
        const uint32_t height = tex.Height(0);
        const GpuFormat fmt = tex.Format();
        const uint32_t num_channels = FormatChannels(fmt);

        const GpuBaseFormat base_fmt = BaseFormat(fmt);
        torch::Dtype data_type;
        switch (base_fmt)
        {
        case GpuBaseFormat::Float:
            data_type = torch::kFloat32;
            break;

        case GpuBaseFormat::Sint:
        case GpuBaseFormat::Uint:
            data_type = torch::kInt32;
            break;

        default:
            Unreachable("Invalid format");
        }

        auto opts = torch::TensorOptions().dtype(data_type);
        torch::Tensor tensor;
        if (uses_cuda_copy_)
        {
            uint64_t fence_val = gpu_system_.ExecuteAndReset(cmd_list);
            this->WaitExternalSemaphore(fence_val);

            MiniCudaRt::ExternalMemory_t ext_mem = this->ImportFromResource(tex);

            MiniCudaRt::ExternalMemoryMipmappedArrayDesc ext_mem_mip_desc{};
            ext_mem_mip_desc.extent = {width, height, 1};
            ext_mem_mip_desc.format_desc = this->FormatDesc(fmt);
            ext_mem_mip_desc.num_levels = 1;
            ext_mem_mip_desc.flags = MiniCudaRt::ArraySurfaceLoadStore;

            MiniCudaRt::MipmappedArray_t cu_mip_array;
            cuda_rt_.ExternalMemoryGetMappedMipmappedArray(&cu_mip_array, ext_mem, &ext_mem_mip_desc);

            MiniCudaRt::Array_t cu_array;
            cuda_rt_.GetMipmappedArrayLevel(&cu_array, cu_mip_array, 0);

            opts = opts.device(torch_device_);
            tensor = torch::empty({1, height, width, num_channels}, opts);

            MiniCudaRt::Memcpy3DParams p{};
            p.src_array = cu_array;
            p.dst_ptr.ptr = tensor.mutable_data_ptr();
            p.dst_ptr.pitch = width * FormatSize(fmt);
            p.dst_ptr.x_size = width;
            p.dst_ptr.y_size = height;
            p.extent = ext_mem_mip_desc.extent;
            p.kind = MiniCudaRt::MemcpyKind::DeviceToDevice;
            cuda_rt_.Memcpy3DAsync(&p, copy_stream_);

            cuda_rt_.FreeMipmappedArray(cu_mip_array);
            cuda_rt_.DestroyExternalMemory(ext_mem);

            ++fence_val;
            this->SignalExternalSemaphore(fence_val);
            gpu_system_.GpuWait(GpuSystem::CmdQueueType::Render, fence_val);
        }
        else
        {
            opts = opts.device(torch::kCPU);
            tensor = torch::empty({1, height, width, num_channels}, opts);
            tex.Readback(gpu_system_, cmd_list, 0, tensor.mutable_data_ptr());
            tensor = tensor.to(torch_device_);
        }

        return tensor;
    }

    MiniCudaRt::ExternalMemory_t GpuDiffRenderTorch::ImportFromResource(const GpuResource& resource)
    {
        ID3D12Device* d3d12_device = gpu_system_.NativeDevice();

        const auto res_desc = resource.NativeResource()->GetDesc();
        const auto alloc_info = d3d12_device->GetResourceAllocationInfo(0, 1, &res_desc);

        MiniCudaRt::ExternalMemoryHandleDesc ext_mem_handle_desc{};
        ext_mem_handle_desc.type = MiniCudaRt::ExternalMemoryHandleType::D3D12Resource;
        ext_mem_handle_desc.handle.win32.handle = resource.SharedHandle();
        ext_mem_handle_desc.size = alloc_info.SizeInBytes;
        ext_mem_handle_desc.flags = MiniCudaRt::ExternalMemoryDedicated;

        MiniCudaRt::ExternalMemory_t ext_mem;
        cuda_rt_.ImportExternalMemory(&ext_mem, &ext_mem_handle_desc);
        return ext_mem;
    }

    void GpuDiffRenderTorch::WaitExternalSemaphore(uint64_t fence_val)
    {
        MiniCudaRt::ExternalSemaphoreWaitParams ext_semaphore_wait_params{};
        ext_semaphore_wait_params.params.fence.value = fence_val;
        cuda_rt_.WaitExternalSemaphoresAsync(&ext_semaphore_, &ext_semaphore_wait_params, 1, copy_stream_);
    }

    void GpuDiffRenderTorch::SignalExternalSemaphore(uint64_t fence_val)
    {
        MiniCudaRt::ExternalSemaphoreSignalParams ext_semaphore_signal_params{};
        ext_semaphore_signal_params.params.fence.value = fence_val;
        cuda_rt_.SignalExternalSemaphoresAsync(&ext_semaphore_, &ext_semaphore_signal_params, 1, copy_stream_);
    }

    MiniCudaRt::ChannelFormatDesc GpuDiffRenderTorch::FormatDesc(GpuFormat format)
    {
        const GpuBaseFormat base_fmt = BaseFormat(format);
        const uint32_t num_channels = FormatChannels(format);

        MiniCudaRt::ChannelFormatKind kind;
        int ch_bytes;
        switch (base_fmt)
        {
        case GpuBaseFormat::Float:
            kind = MiniCudaRt::ChannelFormatKind::Float;
            ch_bytes = sizeof(float);
            break;

        case GpuBaseFormat::Sint:
            kind = MiniCudaRt::ChannelFormatKind::Signed;
            ch_bytes = sizeof(int32_t);
            break;

        case GpuBaseFormat::Uint:
            kind = MiniCudaRt::ChannelFormatKind::Unsigned;
            ch_bytes = sizeof(uint32_t);
            break;

        default:
            Unreachable("Invalid format");
        }

        int channel_size[4] = {};
        for (uint32_t i = 0; i < num_channels; ++i)
        {
            channel_size[i] = ch_bytes * 8;
        }

        return cuda_rt_.CreateChannelDesc(channel_size[0], channel_size[1], channel_size[2], channel_size[3], kind);
    }
} // namespace AIHoloImager
