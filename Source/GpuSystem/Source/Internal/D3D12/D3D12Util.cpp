// Copyright (c) 2024-2025 Minmin Gong
//

#include "D3D12Util.hpp"

#include "Base/Util.hpp"

namespace AIHoloImager
{
    void SetName(ID3D12Object& d3d12_object, std::string_view name)
    {
        std::u16string utf16_name;
        Convert(utf16_name, std::move(name));
        d3d12_object.SetName(name.empty() ? L"" : reinterpret_cast<const wchar_t*>(utf16_name.c_str()));
    }
} // namespace AIHoloImager
