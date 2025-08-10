// Copyright (c) 2025 Minmin Gong
//

#include "MiniCudaRt.hpp"

#include <cassert>

namespace AIHoloImager
{
    MiniCudaRt::MiniCudaRt() : cudart_dll_("cudart64_12.dll")
    {
        if (cudart_dll_)
        {
            cuda_get_device_properties_ = cudart_dll_.Func<CudaGetDeviceProperties_v2>("cudaGetDeviceProperties_v2");
            cuda_create_channel_desc_ = cudart_dll_.Func<CudaCreateChannelDesc>("cudaCreateChannelDesc");
            cuda_import_external_memory_ = cudart_dll_.Func<CudaImportExternalMemory>("cudaImportExternalMemory");
            cuda_external_memory_get_mapped_buffer_ =
                cudart_dll_.Func<CudaExternalMemoryGetMappedBuffer>("cudaExternalMemoryGetMappedBuffer");
            cuda_destroy_external_memory_ = cudart_dll_.Func<CudaDestroyExternalMemory>("cudaDestroyExternalMemory");
            cuda_external_memory_get_mapped_mipmapped_array_ =
                cudart_dll_.Func<CudaExternalMemoryGetMappedMipmappedArray>("cudaExternalMemoryGetMappedMipmappedArray");
            cuda_import_external_semaphore_ = cudart_dll_.Func<CudaImportExternalSemaphore>("cudaImportExternalSemaphore");
            cuda_destroy_external_semaphore_ = cudart_dll_.Func<CudaDestroyExternalSemaphore>("cudaDestroyExternalSemaphore");
            cuda_wait_external_semaphores_async_ = cudart_dll_.Func<CudaWaitExternalSemaphoresAsync_v2>("cudaWaitExternalSemaphoresAsync");
            cuda_signal_external_semaphores_async_ =
                cudart_dll_.Func<CudaSignalExternalSemaphoresAsync_v2>("cudaSignalExternalSemaphoresAsync");
            cuda_stream_create_ = cudart_dll_.Func<CudaStreamCreate>("cudaStreamCreate");
            cuda_stream_destroy_ = cudart_dll_.Func<CudaStreamDestroy>("cudaStreamDestroy");
            cuda_memcpy_async_ = cudart_dll_.Func<CudaMemcpyAsync>("cudaMemcpyAsync");
            cuda_memcpy_3d_async_ = cudart_dll_.Func<CudaMemcpy3DAsync>("cudaMemcpy3DAsync");
            cuda_get_mipmapped_array_level_ = cudart_dll_.Func<CudaGetMipmappedArrayLevel>("cudaGetMipmappedArrayLevel");
            cuda_free_mipmapped_array_ = cudart_dll_.Func<CudaFreeMipmappedArray>("cudaFreeMipmappedArray");
            cuda_free_ = cudart_dll_.Func<CudaFree>("cudaFree");
        }
    }

    MiniCudaRt::~MiniCudaRt() = default;

    MiniCudaRt::MiniCudaRt(MiniCudaRt&& other) noexcept = default;
    MiniCudaRt& MiniCudaRt::operator=(MiniCudaRt&& other) noexcept = default;

    MiniCudaRt::operator bool() const noexcept
    {
        return static_cast<bool>(cudart_dll_);
    }

    MiniCudaRt::Error_t MiniCudaRt::GetDeviceProperties(DeviceProp* prop, int32_t device)
    {
        assert(*this && cuda_get_device_properties_);
        return cuda_get_device_properties_(prop, device);
    }

    MiniCudaRt::ChannelFormatDesc MiniCudaRt::CreateChannelDesc(int32_t x, int32_t y, int32_t z, int32_t w, ChannelFormatKind fmt)
    {
        assert(*this && cuda_create_channel_desc_);
        return cuda_create_channel_desc_(x, y, z, w, fmt);
    }

    MiniCudaRt::Error_t MiniCudaRt::ImportExternalMemory(ExternalMemory_t* ext_mem_out, const ExternalMemoryHandleDesc* mem_handle_desc)
    {
        assert(*this && cuda_import_external_memory_);
        return cuda_import_external_memory_(ext_mem_out, mem_handle_desc);
    }

    MiniCudaRt::Error_t MiniCudaRt::DestroyExternalMemory(ExternalMemory_t ext_mem)
    {
        assert(*this && cuda_destroy_external_memory_);
        return cuda_destroy_external_memory_(ext_mem);
    }

    MiniCudaRt::Error_t MiniCudaRt::ExternalMemoryGetMappedBuffer(
        void** dev_ptr, ExternalMemory_t ext_mem, const ExternalMemoryBufferDesc* buffer_desc)
    {
        assert(*this && cuda_external_memory_get_mapped_buffer_);
        return cuda_external_memory_get_mapped_buffer_(dev_ptr, ext_mem, buffer_desc);
    }

    MiniCudaRt::Error_t MiniCudaRt::ExternalMemoryGetMappedMipmappedArray(
        MipmappedArray_t* mipmap, ExternalMemory_t ext_mem, const ExternalMemoryMipmappedArrayDesc* mipmap_desc)
    {
        assert(*this && cuda_external_memory_get_mapped_mipmapped_array_);
        return cuda_external_memory_get_mapped_mipmapped_array_(mipmap, ext_mem, mipmap_desc);
    }

    MiniCudaRt::Error_t MiniCudaRt::ImportExternalSemaphore(
        ExternalSemaphore_t* ext_sem_out, const ExternalSemaphoreHandleDesc* sem_handle_desc)
    {
        assert(*this && cuda_import_external_semaphore_);
        return cuda_import_external_semaphore_(ext_sem_out, sem_handle_desc);
    }

    MiniCudaRt::Error_t MiniCudaRt::DestroyExternalSemaphore(ExternalSemaphore_t ext_sem)
    {
        assert(*this && cuda_destroy_external_semaphore_);
        return cuda_destroy_external_semaphore_(ext_sem);
    }

    MiniCudaRt::Error_t MiniCudaRt::SignalExternalSemaphoresAsync(
        const ExternalSemaphore_t* ext_sem_array, const ExternalSemaphoreSignalParams* params_array, uint32_t num_ext_sems, Stream_t stream)
    {
        assert(*this && cuda_signal_external_semaphores_async_);
        return cuda_signal_external_semaphores_async_(ext_sem_array, params_array, num_ext_sems, stream);
    }

    MiniCudaRt::Error_t MiniCudaRt::WaitExternalSemaphoresAsync(
        const ExternalSemaphore_t* ext_sem_array, const ExternalSemaphoreWaitParams* params_array, uint32_t num_ext_sems, Stream_t stream)
    {
        assert(*this && cuda_wait_external_semaphores_async_);
        return cuda_wait_external_semaphores_async_(ext_sem_array, params_array, num_ext_sems, stream);
    }

    MiniCudaRt::Error_t MiniCudaRt::StreamCreate(Stream_t* stream_out)
    {
        assert(*this && cuda_stream_create_);
        return cuda_stream_create_(stream_out);
    }

    MiniCudaRt::Error_t MiniCudaRt::StreamDestroy(Stream_t stream)
    {
        assert(*this && cuda_stream_destroy_);
        return cuda_stream_destroy_(stream);
    }

    MiniCudaRt::Error_t MiniCudaRt::MemcpyAsync(void* dst, const void* src, size_t count, MemcpyKind kind, Stream_t stream)
    {
        assert(*this && cuda_memcpy_async_);
        return cuda_memcpy_async_(dst, src, count, kind, stream);
    }

    MiniCudaRt::Error_t MiniCudaRt::Memcpy3DAsync(const Memcpy3DParams* p, Stream_t stream)
    {
        assert(*this && cuda_memcpy_3d_async_);
        return cuda_memcpy_3d_async_(p, stream);
    }

    MiniCudaRt::Error_t MiniCudaRt::GetMipmappedArrayLevel(Array_t* level_array, MipmappedArray_const_t mipmapped_array, uint32_t level)
    {
        assert(*this && cuda_get_mipmapped_array_level_);
        return cuda_get_mipmapped_array_level_(level_array, mipmapped_array, level);
    }

    MiniCudaRt::Error_t MiniCudaRt::FreeMipmappedArray(MipmappedArray_t mipmapped_array)
    {
        assert(*this && cuda_free_mipmapped_array_);
        return cuda_free_mipmapped_array_(mipmapped_array);
    }

    MiniCudaRt::Error_t MiniCudaRt::Free(void* dev_ptr)
    {
        assert(*this && cuda_free_);
        return cuda_free_(dev_ptr);
    }

    std::string CombineFileLine(MiniCudaRt::Error_t err, std::string_view file, uint32_t line)
    {
        std::ostringstream ss;
        ss << "CUDA error of 0x" << std::hex << std::setfill('0') << std::setw(8) << static_cast<uint32_t>(err);
        ss << CombineFileLine(std::move(file), line);
        return ss.str();
    }
} // namespace AIHoloImager
