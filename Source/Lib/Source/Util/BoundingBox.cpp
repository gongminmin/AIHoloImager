// Copyright (c) 2024 Minmin Gong
//

#include "BoundingBox.hpp"

#include <cmath>
#include <cstddef>
#include <limits>

#include <glm/geometric.hpp>
#include <glm/mat3x3.hpp>

namespace
{
    bool SolveCubic(float e, float f, float g, glm::vec3& ev) noexcept
    {
        const float p = f - e * e / 3;
        const float q = g - e * f / 3 + e * e * e * 2 / 27;
        const float h = q * q / 4 + p * p * p / 27;

        if (h > 0)
        {
            ev = {0, 0, 0};
            return false;
        }

        if ((h == 0) && (q == 0))
        {
            ev = {-e / 3, -e / 3, -e / 3};
            return true;
        }

        const float d = std::sqrt(q * q / 4.0f - h);
        const float sgn = d < 0 ? -1.0f : 1.0f;
        const float rc = sgn * std::pow(sgn * d, 1 / 3.0f);

        const float theta = std::acos(-q / (2 * d));
        const float cos_th3 = std::cos(theta / 3);
        const float sin_th3 = std::sqrt(3.0f) * std::sin(theta / 3);
        ev = glm::vec3(2 * cos_th3, -(cos_th3 + sin_th3), -(cos_th3 - sin_th3)) * rc - e / 3;

        return true;
    }

    glm::vec3 CalcEigenVector(const float cov[6], float e) noexcept
    {
        glm::vec3 tmp = {
            cov[1] * cov[4] - cov[2] * (cov[3] - e),
            cov[2] * cov[1] - cov[4] * (cov[0] - e),
            (cov[0] - e) * (cov[3] - e) - cov[1] * cov[1],
        };

        if (glm::all(glm::equal(tmp, glm::zero<glm::vec3>())))
        {
            float f1, f2, f3;
            if ((cov[0] - e != 0) || (cov[1] != 0) || (cov[2] != 0))
            {
                f1 = cov[0] - e;
                f2 = cov[1];
                f3 = cov[2];
            }
            else if ((cov[1] != 0) || (cov[3] - e != 0) || (cov[4] != 0))
            {
                f1 = cov[1];
                f2 = cov[3] - e;
                f3 = cov[4];
            }
            else if ((cov[2] != 0) || (cov[4] != 0) || (cov[5] - e != 0))
            {
                f1 = cov[2];
                f2 = cov[4];
                f3 = cov[5] - e;
            }
            else
            {
                f1 = 1;
                f2 = 0;
                f3 = 0;
            }

            tmp.x = (f1 == 0) ? 0.0f : 1.0f;
            tmp.y = (f2 == 0) ? 0.0f : 1.0f;
            if (f3 == 0)
            {
                tmp.z = 0;
                if (cov[1] != 0)
                {
                    tmp.y = -f1 / f2;
                }
            }
            else
            {
                tmp.z = (f2 - f1) / f3;
            }
        }

        if (glm::dot(tmp, tmp) <= 1e-5f)
        {
            tmp *= 1e5f;
        }

        return glm::normalize(tmp);
    }

    void CalcEigenVectors(const float cov[6], const glm::vec3& ev, glm::mat3x3& mtx) noexcept
    {
        constexpr glm::vec3 Zero = glm::zero<glm::vec3>();

        bool vz[3];
        for (uint32_t i = 0; i < 3; ++i)
        {
            mtx[i] = CalcEigenVector(cov, ev[i]);
            vz[i] = glm::all(glm::equal(mtx[i], Zero));
        }

        const bool e01 = std::abs(glm::dot(mtx[0], mtx[1])) > 0.1f;
        const bool e02 = std::abs(glm::dot(mtx[0], mtx[2])) > 0.1f;
        const bool e12 = std::abs(glm::dot(mtx[1], mtx[2])) > 0.1f;

        if ((vz[0] && vz[1] && vz[2]) || (e01 && e02 && e12) || (e01 && vz[2]) || (e02 && vz[1]) || (e12 && vz[0]))
        {
            mtx = glm::identity<glm::mat3x3>();
            return;
        }

        const auto check_mtx = [&](uint32_t i0, uint32_t i1, uint32_t i2) {
            if (vz[i0] && vz[i1])
            {
                glm::vec3 tmp = glm::cross(glm::vec3(0, 1, 0), mtx[i2]);
                if (glm::dot(tmp, tmp) < 1e-5f)
                {
                    tmp = glm::cross(glm::vec3(1, 0, 0), mtx[i2]);
                }
                mtx[i0] = glm::normalize(tmp);
                mtx[i1] = glm::cross(mtx[i2], mtx[i0]);
                return true;
            }
            return false;
        };

        if (check_mtx(0, 1, 2))
        {
            return;
        }
        if (check_mtx(2, 0, 1))
        {
            return;
        }
        if (check_mtx(1, 2, 0))
        {
            return;
        }

        if (vz[0] || e01)
        {
            mtx[0] = glm::cross(mtx[1], mtx[2]);
            return;
        }
        if (vz[1] || e12)
        {
            mtx[1] = glm::cross(mtx[2], mtx[0]);
            return;
        }
        if (vz[2] || e02)
        {
            mtx[2] = glm::cross(mtx[0], mtx[1]);
            return;
        }
    }

    void CalcEigenVectorsFromCovMatrix(const float cov[6], glm::mat3x3& mtx) noexcept
    {
        const float e = -(cov[0] + cov[3] + cov[5]);
        const float f = cov[0] * cov[3] + cov[3] * cov[5] + cov[5] * cov[0] - cov[1] * cov[1] - cov[2] * cov[2] - cov[4] * cov[4];
        const float g = cov[1] * cov[1] * cov[5] + cov[2] * cov[2] * cov[3] + cov[4] * cov[4] * cov[0] - cov[1] * cov[4] * cov[2] * 2 -
                        cov[0] * cov[3] * cov[5];

        glm::vec3 ev;
        if (SolveCubic(e, f, g, ev))
        {
            CalcEigenVectors(cov, ev, mtx);
        }
        else
        {
            mtx = glm::identity<glm::mat3x3>();
        }
    }
} // namespace

namespace AIHoloImager
{
    Obb Obb::FromPoints(const glm::vec3* positions, uint32_t stride, uint32_t num_vertices)
    {
        auto get_pos = [&](uint32_t index) {
            return *reinterpret_cast<const glm::vec3*>(reinterpret_cast<const std::byte*>(positions) + index * stride);
        };

        glm::vec3 center(0, 0, 0);
        for (uint32_t i = 0; i < num_vertices; ++i)
        {
            center += get_pos(i);
        }
        center /= static_cast<float>(num_vertices);

        float cov[6]{};
        for (uint32_t i = 0; i < num_vertices; ++i)
        {
            const glm::vec3 diff = get_pos(i) - center;
            cov[0] += diff.x * diff.x;
            cov[1] += diff.x * diff.y;
            cov[2] += diff.x * diff.z;
            cov[3] += diff.y * diff.y;
            cov[4] += diff.y * diff.z;
            cov[5] += diff.z * diff.z;
        }

        glm::mat3x3 rot_mtx;
        CalcEigenVectorsFromCovMatrix(cov, rot_mtx);

        if (glm::determinant(rot_mtx) < 0)
        {
            rot_mtx = -rot_mtx;
        }

        Obb obb;
        obb.orientation = glm::normalize(glm::quat_cast(rot_mtx));

        rot_mtx = glm::mat3_cast(obb.orientation);
        const glm::mat3 inv_rot_mtx = glm::transpose(rot_mtx);

        glm::vec3 pmin(std::numeric_limits<float>::max());
        glm::vec3 pmax(std::numeric_limits<float>::lowest());
        for (uint32_t i = 0; i < num_vertices; ++i)
        {
            const glm::vec3 point = inv_rot_mtx * get_pos(i);
            pmin = glm::min(pmin, point);
            pmax = glm::max(pmax, point);
        }

        obb.center = rot_mtx * ((pmin + pmax) * 0.5f);
        obb.extents = (pmax - pmin) * 0.5f;

        return obb;
    }

    Obb Obb::Transform(const Obb& obb, const glm::mat4x4& mtx)
    {
        Obb transformed_obb;

        glm::mat3x3 norm_mtx = mtx;

        const float dx = glm::length(norm_mtx[0]);
        const float dy = glm::length(norm_mtx[1]);
        const float dz = glm::length(norm_mtx[2]);
        transformed_obb.extents = obb.extents * glm::vec3(dx, dy, dz);

        norm_mtx[0] /= dx;
        norm_mtx[1] /= dy;
        norm_mtx[2] /= dz;
        transformed_obb.orientation = glm::quat_cast(norm_mtx) * obb.orientation;

        const glm::vec4 center4 = mtx * glm::vec4(obb.center, 1);
        transformed_obb.center = glm::vec3(center4) / center4.w;

        return transformed_obb;
    }

    void Obb::GetCorners(const Obb& obb, std::span<glm::vec3> corners)
    {
        constexpr glm::vec3 BoxOffsets[] = {
            {-1.0f, -1.0f, +1.0f},
            {+1.0f, -1.0f, +1.0f},
            {+1.0f, +1.0f, +1.0f},
            {-1.0f, +1.0f, +1.0f},
            {-1.0f, -1.0f, -1.0f},
            {+1.0f, -1.0f, -1.0f},
            {+1.0f, +1.0f, -1.0f},
            {-1.0f, +1.0f, -1.0f},
        };

        for (size_t i = 0; i < std::min(corners.size(), std::size(BoxOffsets)); ++i)
        {
            corners[i] = obb.orientation * (obb.extents * BoxOffsets[i]) + obb.center;
        }
    }
} // namespace AIHoloImager
