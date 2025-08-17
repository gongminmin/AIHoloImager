// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include "Base/Dll.hpp"
#include "Base/ErrorHandling.hpp"
#include "Base/Noncopyable.hpp"

// Keep some entries we used from CUDA runtime, so we don't need to reference to CUDA Toolkit

#ifdef _WIN32
    #define CUDART_API __stdcall
#else
    #define CUDART_API
#endif

namespace AIHoloImager
{
    class MiniCudaRt
    {
        DISALLOW_COPY_AND_ASSIGN(MiniCudaRt);

    public:
        enum class Error_t
        {
            Success = 0,
        };

        enum class ChannelFormatKind
        {
            Signed = 0,
            Unsigned,
            Float,
        };

        enum class MemcpyKind
        {
            HostToHost = 0,
            HostToDevice,
            DeviceToHost,
            DeviceToDevice,
            Default,
        };

        enum class ExternalMemoryHandleType
        {
            OpaqueFd = 1,
            OpaqueWin32,
            OpaqueWin32Kmt,
            D3D12Heap,
            D3D12Resource,
            D3D11Resource,
            D3D11ResourceKmt,
            NvSciBuf,
        };

        enum class ExternalSemaphoreHandleType
        {
            OpaqueFd = 1,
            OpaqueWin32,
            OpaqueWin32Kmt,
            D3D12Fence,
            D3D11Fence,
            NvSciSync,
            KeyedMutex,
            KeyedMutexKmt,
            TimelineSemaphoreFd,
            TimelineSemaphoreWin32,
        };

        static constexpr uint32_t ArraySurfaceLoadStore = 0x02;
        static constexpr uint32_t ExternalMemoryDedicated = 0x1;

        using Stream_t = void*;
        using Array_t = void*;
        using MipmappedArray_t = void*;
        using MipmappedArray_const_t = const void*;
        using ExternalMemory_t = void*;
        using ExternalSemaphore_t = void*;

        struct DeviceProp
        {
            char name[256];
            char uuid[16];
            char luid[8];
            uint32_t luid_device_node_mask;
            uint8_t reserved[748];
        };
        static_assert(sizeof(DeviceProp) == 1032);

        struct ChannelFormatDesc
        {
            int32_t x;
            int32_t y;
            int32_t z;
            int32_t w;
            ChannelFormatKind fmt;
        };

        struct Pos
        {
            size_t x;
            size_t y;
            size_t z;
        };

        struct PitchedPtr
        {
            void* ptr;
            size_t pitch;
            size_t x_size;
            size_t y_size;
        };

        struct Extent
        {
            size_t width;
            size_t height;
            size_t depth;
        };

        struct Memcpy3DParams
        {
            Array_t src_array;
            Pos src_pos;
            PitchedPtr src_ptr;

            Array_t dst_array;
            Pos dst_pos;
            PitchedPtr dst_ptr;

            Extent extent;
            MemcpyKind kind;
        };

        struct ExternalMemoryHandleDesc
        {
            ExternalMemoryHandleType type;
            union HandleDesc
            {
                int32_t fd;
                struct Win32HandleDesc
                {
                    void* handle;
                    const void* name;
                } win32;
            } handle;
            uint64_t size;
            uint32_t flags;
        };

        struct ExternalMemoryBufferDesc
        {
            uint64_t offset;
            uint64_t size;
            uint32_t flags;
        };

        struct ExternalMemoryMipmappedArrayDesc
        {
            uint64_t offset;
            ChannelFormatDesc format_desc;
            Extent extent;
            uint32_t flags;
            uint32_t num_levels;
        };

        struct ExternalSemaphoreHandleDesc
        {
            ExternalSemaphoreHandleType type;
            union HandleDesc
            {
                int32_t fd;
                struct Win32HandleDesc
                {
                    void* handle;
                    const void* name;
                } win32;
            } handle;
            uint32_t flags;
        };

        struct ExternalSemaphoreWaitParams
        {
            struct WaitParam
            {
                struct FenceParam
                {
                    uint64_t value;
                } fence;
                union NvSciSyncParam
                {
                    void* fence;
                    uint64_t reserved;
                } nv_sci_sync;
                struct KeyedMutexParam
                {
                    uint64_t key;
                    uint32_t timeout_ms;
                } keyed_mutex;
                uint32_t reserved[10];
            } params;

            uint32_t flags;
            uint32_t reserved[16];
        };

        struct ExternalSemaphoreSignalParams
        {
            struct SignalParam
            {
                struct FenceParam
                {
                    uint64_t value;
                } fence;
                union NvSciSyncParam
                {
                    void* fence;
                    uint64_t reserved;
                } nv_sci_sync;
                struct KeyedMutexParam
                {
                    uint64_t key;
                } keyed_mutex;
                uint32_t reserved[12];
            } params;

            uint32_t flags;
            uint32_t reserved[16];
        };

    public:
        MiniCudaRt();
        MiniCudaRt(MiniCudaRt&& other) noexcept;
        ~MiniCudaRt();

        MiniCudaRt& operator=(MiniCudaRt&& other) noexcept;

        explicit operator bool() const noexcept;

        Error_t GetDeviceProperties(DeviceProp* prop, int32_t device) const noexcept;

        ChannelFormatDesc CreateChannelDesc(int32_t x, int32_t y, int32_t z, int32_t w, ChannelFormatKind fmt) const noexcept;

        Error_t ImportExternalMemory(ExternalMemory_t* ext_mem_out, const ExternalMemoryHandleDesc* mem_handle_desc) const noexcept;
        Error_t DestroyExternalMemory(ExternalMemory_t ext_mem) const noexcept;
        Error_t ExternalMemoryGetMappedBuffer(
            void** dev_ptr, ExternalMemory_t ext_mem, const ExternalMemoryBufferDesc* buffer_desc) const noexcept;
        Error_t ExternalMemoryGetMappedMipmappedArray(
            MipmappedArray_t* mipmap, ExternalMemory_t ext_mem, const ExternalMemoryMipmappedArrayDesc* mipmap_desc) const noexcept;

        Error_t ImportExternalSemaphore(
            ExternalSemaphore_t* ext_sem_out, const ExternalSemaphoreHandleDesc* sem_handle_desc) const noexcept;
        Error_t DestroyExternalSemaphore(ExternalSemaphore_t ext_sem) const noexcept;
        Error_t WaitExternalSemaphoresAsync(const ExternalSemaphore_t* ext_sem_array, const ExternalSemaphoreWaitParams* params_array,
            uint32_t num_ext_sems, Stream_t stream = 0) const noexcept;
        Error_t SignalExternalSemaphoresAsync(const ExternalSemaphore_t* ext_sem_array, const ExternalSemaphoreSignalParams* params_array,
            uint32_t num_ext_sems, Stream_t stream = 0) const noexcept;

        Error_t StreamCreate(Stream_t* stream_out) const noexcept;
        Error_t StreamDestroy(Stream_t stream) const noexcept;

        Error_t MemcpyAsync(void* dst, const void* src, size_t count, MemcpyKind kind, Stream_t stream) const noexcept;
        Error_t Memcpy3DAsync(const Memcpy3DParams* p, Stream_t stream) const noexcept;

        Error_t GetMipmappedArrayLevel(Array_t* level_array, MipmappedArray_const_t mipmapped_array, uint32_t level) const noexcept;
        Error_t FreeMipmappedArray(MipmappedArray_t mipmapped_array) const noexcept;

        Error_t Free(void* dev_ptr) const noexcept;

    private:
        using CudaGetDeviceProperties_v2 = Error_t(CUDART_API*)(DeviceProp* prop, int32_t device) noexcept;
        using CudaCreateChannelDesc = ChannelFormatDesc(CUDART_API*)(
            int32_t x, int32_t y, int32_t z, int32_t w, ChannelFormatKind fmt) noexcept;
        using CudaImportExternalMemory = Error_t(CUDART_API*)(
            ExternalMemory_t* ext_mem_out, const ExternalMemoryHandleDesc* mem_handle_desc) noexcept;
        using CudaDestroyExternalMemory = Error_t(CUDART_API*)(ExternalMemory_t ext_mem) noexcept;
        using CudaExternalMemoryGetMappedBuffer = Error_t(CUDART_API*)(
            void** dev_ptr, ExternalMemory_t ext_mem, const ExternalMemoryBufferDesc* buffer_desc) noexcept;
        using CudaExternalMemoryGetMappedMipmappedArray = Error_t(CUDART_API*)(
            MipmappedArray_t* mipmap, ExternalMemory_t ext_mem, const ExternalMemoryMipmappedArrayDesc* mipmap_desc) noexcept;
        using CudaImportExternalSemaphore = Error_t(CUDART_API*)(
            ExternalSemaphore_t* ext_sem_out, const ExternalSemaphoreHandleDesc* sem_handle_desc) noexcept;
        using CudaDestroyExternalSemaphore = Error_t(CUDART_API*)(ExternalSemaphore_t ext_sem) noexcept;
        using CudaSignalExternalSemaphoresAsync_v2 = Error_t(CUDART_API*)(const ExternalSemaphore_t* ext_sem_array,
            const ExternalSemaphoreSignalParams* params_array, uint32_t num_ext_sems, Stream_t stream) noexcept;
        using CudaWaitExternalSemaphoresAsync_v2 = Error_t(CUDART_API*)(const ExternalSemaphore_t* ext_sem_array,
            const ExternalSemaphoreWaitParams* params_array, uint32_t num_ext_sems, Stream_t stream) noexcept;
        using CudaStreamCreate = Error_t(CUDART_API*)(Stream_t* stream_out) noexcept;
        using CudaStreamDestroy = Error_t(CUDART_API*)(Stream_t stream) noexcept;
        using CudaMemcpyAsync = Error_t(CUDART_API*)(void* dst, const void* src, size_t count, MemcpyKind kind, Stream_t stream) noexcept;
        using CudaMemcpy3DAsync = Error_t(CUDART_API*)(const Memcpy3DParams* p, Stream_t stream) noexcept;
        using CudaGetMipmappedArrayLevel = Error_t(CUDART_API*)(
            Array_t* level_array, MipmappedArray_const_t mipmapped_array, uint32_t level) noexcept;
        using CudaFreeMipmappedArray = Error_t(__stdcall*)(MipmappedArray_t mipmapped_array) noexcept;
        using CudaFree = Error_t(CUDART_API*)(void* dev_ptr) noexcept;

    private:
        Dll cudart_dll_;

        CudaGetDeviceProperties_v2 cuda_get_device_properties_{};
        CudaCreateChannelDesc cuda_create_channel_desc_{};
        CudaImportExternalMemory cuda_import_external_memory_{};
        CudaExternalMemoryGetMappedBuffer cuda_external_memory_get_mapped_buffer_{};
        CudaDestroyExternalMemory cuda_destroy_external_memory_{};
        CudaExternalMemoryGetMappedMipmappedArray cuda_external_memory_get_mapped_mipmapped_array_{};
        CudaImportExternalSemaphore cuda_import_external_semaphore_{};
        CudaDestroyExternalSemaphore cuda_destroy_external_semaphore_{};
        CudaSignalExternalSemaphoresAsync_v2 cuda_signal_external_semaphores_async_{};
        CudaWaitExternalSemaphoresAsync_v2 cuda_wait_external_semaphores_async_{};
        CudaStreamCreate cuda_stream_create_{};
        CudaStreamDestroy cuda_stream_destroy_{};
        CudaMemcpyAsync cuda_memcpy_async_{};
        CudaMemcpy3DAsync cuda_memcpy_3d_async_{};
        CudaGetMipmappedArrayLevel cuda_get_mipmapped_array_level_{};
        CudaFreeMipmappedArray cuda_free_mipmapped_array_{};
        CudaFree cuda_free_{};
    };

    std::string CombineFileLine(MiniCudaRt::Error_t err, std::string_view file, uint32_t line);

    class CudaErrorException : public std::runtime_error
    {
    public:
        CudaErrorException(MiniCudaRt::Error_t err, std::string_view file, uint32_t line)
            : std::runtime_error(CombineFileLine(err, std::move(file), line)), err_(err)
        {
        }

        MiniCudaRt::Error_t Error() const noexcept
        {
            return err_;
        }

    private:
        const MiniCudaRt::Error_t err_;
    };
} // namespace AIHoloImager

#define TIFCE(err)                                                           \
    {                                                                        \
        if (err != MiniCudaRt::Error_t::Success)                             \
        {                                                                    \
            throw AIHoloImager::CudaErrorException(err, __FILE__, __LINE__); \
        }                                                                    \
    }
