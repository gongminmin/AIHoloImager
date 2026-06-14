// Copyright (c) 2025-2026 Minmin Gong
//

#include "TensorConverter/TensorConverter.hpp"

#ifdef _DEBUG
    #undef _DEBUG // Stop linking to python<ver>_d.lib
#endif
#include <Python.h>

#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4018) // Ignore signed/unsigned compare mismatch
    #pragma warning(disable : 4100) // Ignore unreferenced formal parameters
    #pragma warning(disable : 4127) // Ignore constant conditional expression
    #pragma warning(disable : 4244) // Ignore type conversion from `int` to `float`
    #pragma warning(disable : 4251) // Ignore non dll-interface as member
    #pragma warning(disable : 4267) // Ignore type conversion from `size_t` to something else
    #pragma warning(disable : 4324) // Ignore padded structure
    #pragma warning(disable : 4275) // Ignore non dll-interface base class
#endif
#include <torch/types.h>
#ifdef _MSC_VER
    #pragma warning(pop)
#endif
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4251) // Ignore non dll-interface as member
    #pragma warning(disable : 4275) // Ignore non dll-interface base class
#endif
#include <torch/csrc/autograd/python_variable.h>
#ifdef _MSC_VER
    #pragma warning(pop)
#endif

#include "Base/ErrorHandling.hpp"
#include "Gpu/GpuCommandList.hpp"
#include "MiniCudaRt.hpp"

namespace AIHoloImager
{
    class TensorConverter::Impl
    {
    public:
        Impl(GpuSystem& gpu_system, std::string_view torch_device)
            : Impl(gpu_system, torch::Device(TorchDeviceType(std::move(torch_device))))
        {
        }

        Impl(GpuSystem& gpu_system, torch::Device torch_device) : gpu_system_(gpu_system), torch_device_(std::move(torch_device))
        {
            cuda_copy_enabled_ = false;
            if ((torch_device_.type() == torch::DeviceType::CUDA) && cuda_rt_)
            {
                int device_index = 0;
                if (torch_device_.has_index())
                {
                    device_index = torch_device_.index();
                }

                MiniCudaRt::DeviceProp device_prop{};
                TIFCE(cuda_rt_.GetDeviceProperties(&device_prop, device_index));

                const LUID gpu_luid = gpu_system_.DeviceLuid();

                cuda_copy_enabled_ = (std::memcmp(&gpu_luid, device_prop.luid, sizeof(gpu_luid)) == 0);
            }

            for (auto& ext_semaphore : ext_semaphores_)
            {
                ext_semaphore = {};
            }

            if (cuda_copy_enabled_)
            {
                TIFCE(cuda_rt_.StreamCreate(&copy_stream_));
            }
        }

        ~Impl() noexcept
        {
            if (cuda_copy_enabled_)
            {
                for (auto ext_semaphore : ext_semaphores_)
                {
                    if (ext_semaphore != MiniCudaRt::ExternalSemaphore_t{})
                    {
                        cuda_rt_.DestroyExternalSemaphore(ext_semaphore);
                    }
                }
                cuda_rt_.StreamDestroy(copy_stream_);
            }
        }

        static torch::DeviceType TorchDeviceType(std::string_view torch_device)
        {
            // Must match AIHoloImager::Impl::GetDeviceName

            if (torch_device == "cuda")
            {
                return torch::DeviceType::CUDA;
            }
            else
            {
                return torch::DeviceType::CPU;
            }
        }

        void Convert(GpuCommandList& cmd_list, const torch::Tensor& input_tensor, GpuBuffer& buff, GpuHeap heap, GpuResourceFlag flags,
            std::string_view name) const
        {
            const bool uses_cuda_copy = cuda_copy_enabled_ && this->SameTorchDevice(input_tensor.device());
            const uint32_t size = static_cast<uint32_t>(input_tensor.nbytes());

            if (uses_cuda_copy)
            {
                flags |= GpuResourceFlag::Shareable;
            }

            if (buff.Size() != size)
            {
                buff = GpuBuffer(gpu_system_, size, heap, flags);
            }
            buff.Name(std::move(name));

            if (uses_cuda_copy)
            {
                torch::Tensor tensor = input_tensor.contiguous();

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

                copy_buff->Transition(cmd_list, GpuResourceState::CopyDst);

                const auto queue_type = cmd_list.Type();
                this->ImportSemaphore(queue_type);

                uint64_t fence_val = gpu_system_.ExecuteAndReset(cmd_list);
                this->WaitExternalSemaphore(queue_type, fence_val);

                MiniCudaRt::ExternalMemory_t ext_mem = this->ImportFromResource(*copy_buff);

                const MiniCudaRt::ExternalMemoryBufferDesc ext_mem_buffer_desc{
                    .offset = 0,
                    .size = size,
                    .flags = 0,
                };
                void* ext_mem_ptr;
                TIFCE(cuda_rt_.ExternalMemoryGetMappedBuffer(&ext_mem_ptr, ext_mem, &ext_mem_buffer_desc));

                TIFCE(
                    cuda_rt_.MemcpyAsync(ext_mem_ptr, tensor.const_data_ptr(), size, MiniCudaRt::MemcpyKind::DeviceToDevice, copy_stream_));

                TIFCE(cuda_rt_.Free(ext_mem_ptr));
                TIFCE(cuda_rt_.DestroyExternalMemory(ext_mem));

                ++fence_val;
                this->SignalExternalSemaphore(queue_type, fence_val);
                gpu_system_.GpuWait(queue_type, ToWaitFences(std::span<const GpuSystem::WaitQueueFence>({{queue_type, fence_val}})));

                if (heap != GpuHeap::Default)
                {
                    cmd_list.Copy(buff, *copy_buff);
                }
            }
            else
            {
                torch::Tensor tensor = input_tensor.to(torch::kCPU).contiguous();
                cmd_list.Upload(buff, tensor.const_data_ptr(), static_cast<uint32_t>(tensor.nbytes()));
            }
        }

        void Convert(GpuCommandList& cmd_list, const torch::Tensor& input_tensor, GpuTexture& tex, GpuFormat format, GpuResourceFlag flags,
            std::string_view name) const
        {
            assert(input_tensor.size(-1) == FormatChannels(format));
            assert(input_tensor.element_size() == FormatChannelSize(format));

            const bool uses_cuda_copy = cuda_copy_enabled_ && this->SameTorchDevice(input_tensor.device());

            if (uses_cuda_copy)
            {
                flags |= GpuResourceFlag::Shareable;
            }

            const bool is_tex_2d = (input_tensor.sizes().size() == 3) || (input_tensor.size(-4) == 1);
            const uint32_t width = static_cast<uint32_t>(input_tensor.size(-2));
            const uint32_t height = static_cast<uint32_t>(input_tensor.size(-3));
            const uint32_t depth = is_tex_2d ? 1 : static_cast<uint32_t>(input_tensor.size(-4));

            bool recreate = false;
            if ((tex.Width(0) != width) || (tex.Height(0) != height) || (tex.Format() != format))
            {
                recreate = true;
            }
            if (!is_tex_2d && (tex.Depth(0) != depth))
            {
                recreate = true;
            }

            if (recreate)
            {
                if (is_tex_2d)
                {
                    tex = GpuTexture2D(gpu_system_, width, height, 1, format, flags);
                }
                else
                {
                    tex = GpuTexture3D(gpu_system_, width, height, depth, 1, format, flags);
                }
            }
            tex.Name(std::move(name));

            if (uses_cuda_copy)
            {
                tex.Transition(cmd_list, GpuResourceState::CopyDst);

                const auto queue_type = cmd_list.Type();
                this->ImportSemaphore(queue_type);

                uint64_t fence_val = gpu_system_.ExecuteAndReset(cmd_list);
                this->WaitExternalSemaphore(queue_type, fence_val);

                torch::Tensor tensor = input_tensor.contiguous();

                MiniCudaRt::ExternalMemory_t ext_mem = this->ImportFromResource(tex);

                MiniCudaRt::ExternalMemoryMipmappedArrayDesc ext_mem_mip_desc{
                    .format_desc = this->FormatDesc(format),
                    .extent{
                        .width = width,
                        .height = height,
                        .depth = is_tex_2d ? 0 : depth,
                    },
                    .flags = MiniCudaRt::ArraySurfaceLoadStore,
                    .num_levels = 1,
                };
                MiniCudaRt::MipmappedArray_t cu_mip_array;
                TIFCE(cuda_rt_.ExternalMemoryGetMappedMipmappedArray(&cu_mip_array, ext_mem, &ext_mem_mip_desc));

                MiniCudaRt::Array_t cu_array;
                TIFCE(cuda_rt_.GetMipmappedArrayLevel(&cu_array, cu_mip_array, 0));

                const MiniCudaRt::Memcpy3DParams param{
                    .src_ptr{
                        .ptr = tensor.mutable_data_ptr(),
                        .pitch = width * FormatSize(format),
                        .x_size = width,
                        .y_size = height,
                    },
                    .dst_array = cu_array,
                    .extent{
                        .width = width,
                        .height = height,
                        .depth = depth,
                    },
                    .kind = MiniCudaRt::MemcpyKind::DeviceToDevice,
                };
                TIFCE(cuda_rt_.Memcpy3DAsync(&param, copy_stream_));

                TIFCE(cuda_rt_.FreeMipmappedArray(cu_mip_array));
                TIFCE(cuda_rt_.DestroyExternalMemory(ext_mem));

                ++fence_val;
                this->SignalExternalSemaphore(queue_type, fence_val);
                gpu_system_.GpuWait(queue_type, ToWaitFences(std::span<const GpuSystem::WaitQueueFence>({{queue_type, fence_val}})));
            }
            else
            {
                torch::Tensor tensor = input_tensor.to(torch::kCPU).contiguous();
                cmd_list.Upload(tex, 0, tensor.const_data_ptr(), static_cast<uint32_t>(tensor.nbytes()));
            }
        }

        torch::Tensor Convert(GpuCommandList& cmd_list, const GpuBuffer& buff, std::span<const int64_t> size, DataType data_type) const
        {
            torch::Dtype torch_dtype = torch::kFloat32;
            switch (data_type)
            {
            case DataType::Int32:
                torch_dtype = torch::kInt32;
                break;
            case DataType::Float32:
                torch_dtype = torch::kFloat32;
                break;
            }

            auto opts = torch::dtype(torch_dtype);
            torch::Tensor tensor;
            if (cuda_copy_enabled_)
            {
                buff.Transition(cmd_list, GpuResourceState::CopySrc);

                const auto queue_type = cmd_list.Type();
                this->ImportSemaphore(queue_type);

                uint64_t fence_val = gpu_system_.ExecuteAndReset(cmd_list);
                this->WaitExternalSemaphore(queue_type, fence_val);

                MiniCudaRt::ExternalMemory_t ext_mem = this->ImportFromResource(buff);

                const MiniCudaRt::ExternalMemoryBufferDesc ext_mem_buffer_desc{
                    .offset = 0,
                    .size = buff.Size(),
                    .flags = 0,
                };
                void* ext_mem_ptr;
                TIFCE(cuda_rt_.ExternalMemoryGetMappedBuffer(&ext_mem_ptr, ext_mem, &ext_mem_buffer_desc));

                opts = opts.device(torch_device_);
                tensor = torch::empty(size, opts);

                TIFCE(cuda_rt_.MemcpyAsync(
                    tensor.mutable_data_ptr(), ext_mem_ptr, tensor.nbytes(), MiniCudaRt::MemcpyKind::DeviceToDevice, copy_stream_));

                TIFCE(cuda_rt_.Free(ext_mem_ptr));
                TIFCE(cuda_rt_.DestroyExternalMemory(ext_mem));

                ++fence_val;
                this->SignalExternalSemaphore(queue_type, fence_val);
                gpu_system_.GpuWait(queue_type, ToWaitFences(std::span<const GpuSystem::WaitQueueFence>({{queue_type, fence_val}})));
            }
            else
            {
                opts = opts.device(torch::kCPU);
                tensor = torch::empty(size, opts);
                const auto rb_future = cmd_list.ReadBackAsync(buff, tensor.mutable_data_ptr(), static_cast<uint32_t>(tensor.nbytes()));
                rb_future.wait();
                if (torch_device_.type() != torch::DeviceType::CPU)
                {
                    tensor = tensor.to(torch_device_);
                }
            }

            return tensor;
        }

        torch::Tensor Convert(GpuCommandList& cmd_list, const GpuTexture& tex) const
        {
            const bool is_tex_2d = (tex.Type() == GpuResourceType::Texture2D);
            const uint32_t width = tex.Width(0);
            const uint32_t height = tex.Height(0);
            const uint32_t depth = tex.Depth(0);
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

            case GpuBaseFormat::UNorm:
                data_type = torch::kUInt8;
                break;

            default:
                Unreachable("Invalid format");
            }

            auto opts = torch::dtype(data_type);
            torch::Tensor tensor;
            if (cuda_copy_enabled_)
            {
                tex.Transition(cmd_list, GpuResourceState::CopySrc);

                const auto queue_type = cmd_list.Type();
                this->ImportSemaphore(queue_type);

                uint64_t fence_val = gpu_system_.ExecuteAndReset(cmd_list);
                this->WaitExternalSemaphore(queue_type, fence_val);

                MiniCudaRt::ExternalMemory_t ext_mem = this->ImportFromResource(tex);

                const MiniCudaRt::ExternalMemoryMipmappedArrayDesc ext_mem_mip_desc{
                    .format_desc = this->FormatDesc(fmt),
                    .extent{
                        .width = width,
                        .height = height,
                        .depth = is_tex_2d ? 0 : depth,
                    },
                    .flags = MiniCudaRt::ArraySurfaceLoadStore,
                    .num_levels = 1,
                };
                MiniCudaRt::MipmappedArray_t cu_mip_array;
                TIFCE(cuda_rt_.ExternalMemoryGetMappedMipmappedArray(&cu_mip_array, ext_mem, &ext_mem_mip_desc));

                MiniCudaRt::Array_t cu_array;
                TIFCE(cuda_rt_.GetMipmappedArrayLevel(&cu_array, cu_mip_array, 0));

                opts = opts.device(torch_device_);
                tensor = torch::empty({depth, height, width, num_channels}, opts);

                const MiniCudaRt::Memcpy3DParams param{
                    .src_array = cu_array,
                    .dst_ptr{
                        .ptr = tensor.mutable_data_ptr(),
                        .pitch = width * FormatSize(fmt),
                        .x_size = width,
                        .y_size = height,
                    },
                    .extent{
                        .width = width,
                        .height = height,
                        .depth = depth,
                    },
                    .kind = MiniCudaRt::MemcpyKind::DeviceToDevice,
                };
                TIFCE(cuda_rt_.Memcpy3DAsync(&param, copy_stream_));

                TIFCE(cuda_rt_.FreeMipmappedArray(cu_mip_array));
                TIFCE(cuda_rt_.DestroyExternalMemory(ext_mem));

                ++fence_val;
                this->SignalExternalSemaphore(queue_type, fence_val);
                gpu_system_.GpuWait(queue_type, ToWaitFences(std::span<const GpuSystem::WaitQueueFence>({{queue_type, fence_val}})));
            }
            else
            {
                opts = opts.device(torch::kCPU);
                tensor = torch::empty({depth, height, width, num_channels}, opts);
                const auto rb_future = cmd_list.ReadBackAsync(tex, 0, tensor.mutable_data_ptr(), static_cast<uint32_t>(tensor.nbytes()));
                rb_future.wait();
                if (torch_device_.type() != torch::DeviceType::CPU)
                {
                    tensor = tensor.to(torch_device_);
                }
            }

            return tensor;
        }

        void ConvertPy(GpuCommandList& cmd_list, const PyObject& py_tensor, GpuBuffer& buff, GpuHeap heap, GpuResourceFlag flags,
            std::string_view name) const
        {
            const torch::Tensor& tensor = THPVariable_Unpack(const_cast<PyObject*>(&py_tensor));
            this->Convert(cmd_list, tensor, buff, heap, flags, std::move(name));
        }

        void ConvertPy(GpuCommandList& cmd_list, const PyObject& py_tensor, GpuTexture& tex, GpuFormat format, GpuResourceFlag flags,
            std::string_view name) const
        {
            const torch::Tensor& tensor = THPVariable_Unpack(const_cast<PyObject*>(&py_tensor));
            this->Convert(cmd_list, tensor, tex, format, flags, std::move(name));
        }

        PyObject* ConvertPy(GpuCommandList& cmd_list, const GpuBuffer& buff, std::span<const int64_t> size, DataType data_type) const
        {
            const torch::Tensor tensor = this->Convert(cmd_list, buff, std::move(size), data_type);
            return THPVariable_Wrap(tensor);
        }

        PyObject* ConvertPy(GpuCommandList& cmd_list, const GpuTexture& tex) const
        {
            const torch::Tensor tensor = this->Convert(cmd_list, tex);
            return THPVariable_Wrap(tensor);
        }

    private:
        MiniCudaRt::ExternalMemory_t ImportFromResource(const GpuResource& resource) const
        {
            MiniCudaRt::ExternalMemoryHandleDesc ext_mem_handle_desc{
                .handle{
                    .win32{
                        .handle = resource.SharedHandle(),
                    },
                },
                .size = resource.AllocationSize(),
            };
            switch (gpu_system_.NativeApi())
            {
            case GpuSystem::Api::D3D12:
                ext_mem_handle_desc.type = MiniCudaRt::ExternalMemoryHandleType::D3D12Resource;
                ext_mem_handle_desc.flags = MiniCudaRt::ExternalMemoryDedicated;
                break;
            case GpuSystem::Api::Vulkan:
                ext_mem_handle_desc.type = MiniCudaRt::ExternalMemoryHandleType::OpaqueWin32;
                ext_mem_handle_desc.flags = 0;
                break;

            default:
                Unreachable("Invalid API");
            }

            MiniCudaRt::ExternalMemory_t ext_mem;
            TIFCE(cuda_rt_.ImportExternalMemory(&ext_mem, &ext_mem_handle_desc));
            return ext_mem;
        }

        void ImportSemaphore(GpuSystem::CmdQueueType type) const
        {
            if (ext_semaphores_[static_cast<uint32_t>(type)] == MiniCudaRt::ExternalSemaphore_t{})
            {
                MiniCudaRt::ExternalSemaphoreHandleDesc ext_semaphore_handle_desc{
                    .handle{
                        .win32{
                            .handle = gpu_system_.SharedFenceHandle(type),
                        },
                    },
                    .flags = 0,
                };
                switch (gpu_system_.NativeApi())
                {
                case GpuSystem::Api::D3D12:
                    ext_semaphore_handle_desc.type = MiniCudaRt::ExternalSemaphoreHandleType::D3D12Fence;
                    break;
                case GpuSystem::Api::Vulkan:
                    ext_semaphore_handle_desc.type = MiniCudaRt::ExternalSemaphoreHandleType::TimelineSemaphoreWin32;
                    break;

                default:
                    Unreachable("Invalid API");
                }
                TIFCE(cuda_rt_.ImportExternalSemaphore(&ext_semaphores_[static_cast<uint32_t>(type)], &ext_semaphore_handle_desc));
            }
        }

        void WaitExternalSemaphore(GpuSystem::CmdQueueType type, uint64_t fence_val) const
        {
            const MiniCudaRt::ExternalSemaphoreWaitParams ext_semaphore_wait_params{
                .params{
                    .fence{
                        .value = fence_val,
                    },
                },
            };
            TIFCE(cuda_rt_.WaitExternalSemaphoresAsync(
                &ext_semaphores_[static_cast<uint32_t>(type)], &ext_semaphore_wait_params, 1, copy_stream_));
        }

        void SignalExternalSemaphore(GpuSystem::CmdQueueType type, uint64_t fence_val) const
        {
            const MiniCudaRt::ExternalSemaphoreSignalParams ext_semaphore_signal_params{
                .params{
                    .fence{
                        .value = fence_val,
                    },
                },
            };
            TIFCE(cuda_rt_.SignalExternalSemaphoresAsync(
                &ext_semaphores_[static_cast<uint32_t>(type)], &ext_semaphore_signal_params, 1, copy_stream_));
        }

        MiniCudaRt::ChannelFormatDesc FormatDesc(GpuFormat format) const noexcept
        {
            const GpuBaseFormat base_fmt = BaseFormat(format);
            const uint32_t num_channels = FormatChannels(format);
            const uint32_t fmt_ch_size = FormatChannelSize(format);

            MiniCudaRt::ChannelFormatKind kind;
            int ch_bytes;
            switch (base_fmt)
            {
            case GpuBaseFormat::Float:
                kind = MiniCudaRt::ChannelFormatKind::Float;
                if (fmt_ch_size == 2)
                {
                    ch_bytes = sizeof(uint16_t);
                }
                else
                {
                    ch_bytes = sizeof(float);
                }
                break;

            case GpuBaseFormat::Sint:
                kind = MiniCudaRt::ChannelFormatKind::Signed;
                if (fmt_ch_size == 2)
                {
                    ch_bytes = sizeof(int16_t);
                }
                else
                {
                    ch_bytes = sizeof(int32_t);
                }
                break;

            case GpuBaseFormat::Uint:
                kind = MiniCudaRt::ChannelFormatKind::Unsigned;
                if (fmt_ch_size == 2)
                {
                    ch_bytes = sizeof(uint16_t);
                }
                else
                {
                    ch_bytes = sizeof(uint32_t);
                }
                break;

            case GpuBaseFormat::UNorm:
                kind = MiniCudaRt::ChannelFormatKind::Unsigned;
                ch_bytes = sizeof(uint8_t);
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

        bool SameTorchDevice(const torch::Device& device) const noexcept
        {
            if (torch_device_.type() == device.type())
            {
                int init_device_index = 0;
                if (torch_device_.has_index())
                {
                    init_device_index = torch_device_.index();
                }

                int device_index = 0;
                if (device.has_index())
                {
                    device_index = device.index();
                }

                return init_device_index == device_index;
            }

            return false;
        }

    private:
        GpuSystem& gpu_system_;
        torch::Device torch_device_;

        MiniCudaRt cuda_rt_;

        bool cuda_copy_enabled_;
        mutable MiniCudaRt::ExternalSemaphore_t ext_semaphores_[static_cast<uint32_t>(GpuSystem::CmdQueueType::Num)];
        MiniCudaRt::Stream_t copy_stream_{};
    };

    TensorConverter::TensorConverter(GpuSystem& gpu_system, std::string_view torch_device)
        : impl_(std::make_unique<Impl>(gpu_system, std::move(torch_device)))
    {
    }

    TensorConverter::TensorConverter(GpuSystem& gpu_system, const torch::Device& torch_device)
        : impl_(std::make_unique<Impl>(gpu_system, torch_device))
    {
    }

    TensorConverter::~TensorConverter() noexcept = default;

    void TensorConverter::Convert(GpuCommandList& cmd_list, const torch::Tensor& tensor, GpuBuffer& buff, GpuHeap heap,
        GpuResourceFlag flags, std::string_view name) const
    {
        impl_->Convert(cmd_list, tensor, buff, heap, flags, std::move(name));
    }

    void TensorConverter::Convert(GpuCommandList& cmd_list, const torch::Tensor& tensor, GpuTexture& tex, GpuFormat format,
        GpuResourceFlag flags, std::string_view name) const
    {
        impl_->Convert(cmd_list, tensor, tex, format, flags, std::move(name));
    }

    torch::Tensor TensorConverter::Convert(
        GpuCommandList& cmd_list, const GpuBuffer& buff, std::span<const int64_t> size, DataType data_type) const
    {
        return impl_->Convert(cmd_list, buff, std::move(size), data_type);
    }

    torch::Tensor TensorConverter::Convert(GpuCommandList& cmd_list, const GpuTexture& tex) const
    {
        return impl_->Convert(cmd_list, tex);
    }

    void TensorConverter::ConvertPy(GpuCommandList& cmd_list, const PyObject& py_tensor, GpuBuffer& buff, GpuHeap heap,
        GpuResourceFlag flags, std::string_view name) const
    {
        impl_->ConvertPy(cmd_list, py_tensor, buff, heap, flags, std::move(name));
    }

    void TensorConverter::ConvertPy(GpuCommandList& cmd_list, const PyObject& py_tensor, GpuTexture& tex, GpuFormat format,
        GpuResourceFlag flags, std::string_view name) const
    {
        impl_->ConvertPy(cmd_list, py_tensor, tex, format, flags, std::move(name));
    }

    PyObject* TensorConverter::ConvertPy(
        GpuCommandList& cmd_list, const GpuBuffer& buff, std::span<const int64_t> size, DataType data_type) const
    {
        return impl_->ConvertPy(cmd_list, buff, std::move(size), data_type);
    }

    PyObject* TensorConverter::ConvertPy(GpuCommandList& cmd_list, const GpuTexture& tex) const
    {
        return impl_->ConvertPy(cmd_list, tex);
    }
} // namespace AIHoloImager
