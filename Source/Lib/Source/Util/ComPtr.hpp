// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <cassert>
#include <utility>

#include <Util/ErrorHandling.hpp>
#include <Util/Uuid.hpp>

namespace AIHoloImager
{
    template <typename T>
    class ComPtr
    {
        template <typename U>
        friend class ComPtr;

    public:
        using element_type = std::remove_extent_t<T>;

    public:
        ComPtr() noexcept = default;

        ComPtr(std::nullptr_t) noexcept
        {
        }

        ComPtr(T* ptr, bool add_ref = true) noexcept : ptr_(ptr)
        {
            if (add_ref)
            {
                this->InternalAddRef();
            }
        }

        template <typename U>
        ComPtr(U* ptr, bool add_ref = true) noexcept : ptr_(ptr)
        {
            if (add_ref)
            {
                this->InternalAddRef();
            }
        }

        ComPtr(const ComPtr& rhs) noexcept : ComPtr(rhs.ptr_, true)
        {
        }

        template <typename U>
        ComPtr(const ComPtr<U>& rhs) noexcept : ComPtr(rhs.ptr_, true)
        {
        }

        ComPtr(ComPtr&& rhs) noexcept : ptr_(std::exchange(rhs.ptr_, {}))
        {
        }

        template <typename U>
        ComPtr(ComPtr<U>&& rhs) noexcept : ptr_(std::exchange(rhs.ptr_, {}))
        {
        }

        ~ComPtr() noexcept
        {
            this->InternalRelease();
        }

        ComPtr& operator=(const ComPtr& rhs) noexcept
        {
            if (ptr_ != rhs.ptr_)
            {
                this->InternalRelease();
                ptr_ = rhs.ptr_;
                this->InternalAddRef();
            }
            return *this;
        }

        template <typename U>
        ComPtr& operator=(const ComPtr<U>& rhs) noexcept
        {
            if (ptr_ != rhs.ptr_)
            {
                this->InternalRelease();
                ptr_ = rhs.ptr_;
                this->InternalAddRef();
            }
            return *this;
        }

        ComPtr& operator=(ComPtr&& rhs) noexcept
        {
            if (ptr_ != rhs.ptr_)
            {
                this->InternalRelease();
                ptr_ = std::exchange(rhs.ptr_, {});
            }
            return *this;
        }

        template <typename U>
        ComPtr& operator=(ComPtr<U>&& rhs) noexcept
        {
            this->InternalRelease();
            ptr_ = std::exchange(rhs.ptr_, {});
            return *this;
        }

        void Swap(ComPtr& rhs) noexcept
        {
            std::swap(ptr_, rhs.ptr_);
        }

        explicit operator bool() const noexcept
        {
            return (ptr_ != nullptr);
        }

        T& operator*() const noexcept
        {
            return *ptr_;
        }

        T* operator->() const noexcept
        {
            return ptr_;
        }

        T* Get() const noexcept
        {
            return ptr_;
        }

        T** Put() noexcept
        {
            assert(ptr_ == nullptr);
            return &ptr_;
        }

        void** PutVoid() noexcept
        {
            return reinterpret_cast<void**>(this->Put());
        }

        T** ReleaseAndPut() noexcept
        {
            this->Reset();
            return this->put();
        }

        void** ReleaseAndPutVoid() noexcept
        {
            return reinterpret_cast<void**>(this->release_and_put());
        }

        T* Detach() noexcept
        {
            return std::exchange(ptr_, {});
        }

        void Reset() noexcept
        {
            this->InternalRelease();
        }

        void Reset(T* rhs, bool add_ref = true) noexcept
        {
            this->Reset();
            ptr_ = rhs;
            if (add_ref)
            {
                this->InternalAddRef();
            }
        }

        template <typename U>
        ComPtr<U> TryAs() const noexcept
        {
            ComPtr<U> ret;
            ptr_->QueryInterface(UuidOf<U>(), ret.PutVoid());
            return ret;
        }

        template <typename U>
        bool TryAs(ComPtr<U>& to) const noexcept
        {
            to = this->TryAs<U>();
            return static_cast<bool>(to);
        }

        template <typename U>
        ComPtr<U> As() const
        {
            ComPtr<U> ret;
            TIFHR(ptr_->QueryInterface(UuidOf<U>(), ret.PutVoid()));
            return ret;
        }

        template <typename U>
        void As(ComPtr<U>& to) const
        {
            to = this->As<U>();
        }

    private:
        void InternalAddRef() noexcept
        {
            if (ptr_ != nullptr)
            {
                ptr_->AddRef();
            }
        }

        void InternalRelease() noexcept
        {
            if (ptr_ != nullptr)
            {
                std::exchange(ptr_, {})->Release();
            }
        }

    private:
        T* ptr_ = nullptr;
    };

    template <typename T, typename U>
    bool operator==(const ComPtr<T>& lhs, const ComPtr<U>& rhs) noexcept
    {
        return lhs.get() == rhs.get();
    }

    template <typename T>
    bool operator==(const ComPtr<T>& lhs, [[maybe_unused]] std::nullptr_t rhs) noexcept
    {
        return lhs.Get() == nullptr;
    }

    template <typename T>
    bool operator==(std::nullptr_t lhs, const ComPtr<T>& rhs) noexcept
    {
        return rhs == lhs;
    }

    template <typename T, typename U>
    bool operator!=(const ComPtr<T>& lhs, const ComPtr<U>& rhs) noexcept
    {
        return !(lhs == rhs);
    }

    template <typename T>
    bool operator!=(const ComPtr<T>& lhs, std::nullptr_t rhs) noexcept
    {
        return !(lhs == rhs);
    }

    template <typename T>
    bool operator!=(std::nullptr_t lhs, const ComPtr<T>& rhs) noexcept
    {
        return !(lhs == rhs);
    }

    template <typename T, typename U>
    bool operator<(const ComPtr<T>& lhs, const ComPtr<U>& rhs) noexcept
    {
        return lhs.get() < rhs.get();
    }

    template <typename T>
    bool operator<(const ComPtr<T>& lhs, [[maybe_unused]] std::nullptr_t rhs) noexcept
    {
        return lhs.get() < nullptr;
    }

    template <typename T>
    bool operator<(std::nullptr_t lhs, const ComPtr<T>& rhs) noexcept
    {
        return rhs > lhs;
    }

    template <typename T, typename U>
    bool operator<=(const ComPtr<T>& lhs, const ComPtr<U>& rhs) noexcept
    {
        return !(rhs < lhs);
    }

    template <typename T>
    bool operator<=(const ComPtr<T>& lhs, std::nullptr_t rhs) noexcept
    {
        return !(rhs < lhs);
    }

    template <typename T>
    bool operator<=(std::nullptr_t lhs, const ComPtr<T>& rhs) noexcept
    {
        return !(rhs < lhs);
    }

    template <typename T, typename U>
    bool operator>(const ComPtr<T>& lhs, const ComPtr<U>& rhs) noexcept
    {
        return rhs < lhs;
    }

    template <typename T>
    bool operator>(const ComPtr<T>& lhs, std::nullptr_t rhs) noexcept
    {
        return rhs < lhs;
    }

    template <typename T>
    bool operator>(std::nullptr_t lhs, const ComPtr<T>& rhs) noexcept
    {
        return rhs < lhs;
    }

    template <typename T, typename U>
    bool operator>=(const ComPtr<T>& lhs, const ComPtr<U>& rhs) noexcept
    {
        return !(lhs < rhs);
    }

    template <typename T>
    bool operator>=(const ComPtr<T>& lhs, std::nullptr_t rhs) noexcept
    {
        return !(lhs < rhs);
    }

    template <typename T>
    bool operator>=(std::nullptr_t lhs, const ComPtr<T>& rhs) noexcept
    {
        return !(lhs < rhs);
    }
} // namespace AIHoloImager

namespace std
{
    template <typename T>
    void swap(AIHoloImager::ComPtr<T>& lhs, AIHoloImager::ComPtr<T>& rhs) noexcept
    {
        lhs.Swap(rhs);
    }

    template <typename T>
    struct hash<AIHoloImager::ComPtr<T>>
    {
        using argument_type = AIHoloImager::ComPtr<T>;
        using result_type = std::size_t;

        result_type operator()(const argument_type& p) const noexcept
        {
            return std::hash<typename argument_type::element_type*>()(p.get());
        }
    };
} // namespace std
