// Copyright (c) 2025 Minmin Gong
//

#include "Base/Util.hpp"

namespace AIHoloImager
{
    void Convert(std::string& dest, std::u16string_view src)
    {
        dest.clear();
        dest.reserve(src.size() * 3 / 2);

        for (size_t i = 0; i < src.size(); ++i)
        {
            char32_t codepoint = 0;
            const char16_t high = src[i];

            if ((high >= 0xD800) && (high <= 0xDBFF))
            {
                // High surrogate
                if (i + 1 >= src.size())
                {
                    break; // incomplete pair
                }
                char16_t low = src[i + 1];
                if (low < 0xDC00 || low > 0xDFFF)
                {
                    break; // Invalid
                }
                codepoint = 0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00);
                ++i; // consume low surrogate
            }
            else if ((high >= 0xDC00) && (high <= 0xDFFF))
            {
                // Unexpected low surrogate ¡ª skip or handle error
                continue;
            }
            else
            {
                codepoint = high;
            }

            // Encode codepoint to UTF-8
            if (codepoint <= 0x7F)
            {
                dest.push_back(static_cast<char8_t>(codepoint));
            }
            else if (codepoint <= 0x7FF)
            {
                dest.push_back(static_cast<char8_t>(0xC0 | (codepoint >> 6)));
                dest.push_back(static_cast<char8_t>(0x80 | (codepoint & 0x3F)));
            }
            else if (codepoint <= 0xFFFF)
            {
                dest.push_back(static_cast<char8_t>(0xE0 | (codepoint >> 12)));
                dest.push_back(static_cast<char8_t>(0x80 | ((codepoint >> 6) & 0x3F)));
                dest.push_back(static_cast<char8_t>(0x80 | (codepoint & 0x3F)));
            }
            else if (codepoint <= 0x10FFFF)
            {
                dest.push_back(static_cast<char8_t>(0xF0 | (codepoint >> 18)));
                dest.push_back(static_cast<char8_t>(0x80 | ((codepoint >> 12) & 0x3F)));
                dest.push_back(static_cast<char8_t>(0x80 | ((codepoint >> 6) & 0x3F)));
                dest.push_back(static_cast<char8_t>(0x80 | (codepoint & 0x3F)));
            }
        }
    }

    void Convert(std::string& dest, std::string_view src)
    {
        dest = std::string(src);
    }

    void Convert(std::u16string& dest, std::string_view src)
    {
        dest.clear();
        dest.reserve(src.size());

        for (size_t i = 0; i < src.size();)
        {
            char32_t codepoint = 0;
            const char8_t c = src[i];

            if (c <= 0x7F)
            {
                codepoint = c;
                ++i;
            }
            else if ((c & 0xE0) == 0xC0)
            {
                // 2-byte
                if (i + 1 >= src.size())
                    break; // incomplete
                codepoint = ((c & 0x1F) << 6) | (src[i + 1] & 0x3F);
                i += 2;
            }
            else if ((c & 0xF0) == 0xE0)
            {
                // 3-byte
                if (i + 2 >= src.size())
                {
                    break;
                }
                codepoint = ((c & 0x0F) << 12) | ((src[i + 1] & 0x3F) << 6) | (src[i + 2] & 0x3F);
                i += 3;
            }
            else if ((c & 0xF8) == 0xF0)
            {
                // 4-byte
                if (i + 3 >= src.size())
                {
                    break;
                }
                codepoint = ((c & 0x07) << 18) | ((src[i + 1] & 0x3F) << 12) | ((src[i + 2] & 0x3F) << 6) | (src[i + 3] & 0x3F);
                i += 4;
            }
            else
            {
                ++i; // Invalid, skip
                continue;
            }

            if (codepoint <= 0xFFFF)
            {
                dest.push_back(static_cast<char16_t>(codepoint));
            }
            else if (codepoint <= 0x10FFFF)
            {
                codepoint -= 0x10000;
                dest.push_back(static_cast<char16_t>(0xD800 + (codepoint >> 10)));
                dest.push_back(static_cast<char16_t>(0xDC00 + (codepoint & 0x3FF)));
            }
        }
    }

    void Convert(std::u16string& dest, std::u16string_view src)
    {
        dest = std::u16string(src);
    }
} // namespace AIHoloImager
