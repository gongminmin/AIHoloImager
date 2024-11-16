// Copyright (c) 2024 Minmin Gong
//

#include "MeshSimplification.hpp"

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

// QEM-based mesh simplification.
// Based on https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification/blob/master/src.cmd/Simplify.h

namespace AIHoloImager
{
    class SymmetricMatrix
    {
    public:
        SymmetricMatrix()
        {
            for (double& v : m_)
            {
                v = 0;
            }
        }

        SymmetricMatrix(
            double m11, double m12, double m13, double m14, double m22, double m23, double m24, double m33, double m34, double m44)
        {
            m_[0] = m11;
            m_[1] = m12;
            m_[2] = m13;
            m_[3] = m14;
            m_[4] = m22;
            m_[5] = m23;
            m_[6] = m24;
            m_[7] = m33;
            m_[8] = m34;
            m_[9] = m44;
        }

        SymmetricMatrix(double a, double b, double c, double d)
        {
            m_[0] = a * a;
            m_[1] = a * b;
            m_[2] = a * c;
            m_[3] = a * d;
            m_[4] = b * b;
            m_[5] = b * c;
            m_[6] = b * d;
            m_[7] = c * c;
            m_[8] = c * d;
            m_[9] = d * d;
        }

        double operator[](size_t c) const
        {
            return m_[c];
        }

        double Det(size_t a11, size_t a12, size_t a13, size_t a21, size_t a22, size_t a23, size_t a31, size_t a32, size_t a33) const
        {
            return m_[a11] * m_[a22] * m_[a33] + m_[a13] * m_[a21] * m_[a32] + m_[a12] * m_[a23] * m_[a31] - m_[a13] * m_[a22] * m_[a31] -
                   m_[a11] * m_[a23] * m_[a32] - m_[a12] * m_[a21] * m_[a33];
        }

        SymmetricMatrix& operator+=(const SymmetricMatrix& rhs)
        {
            m_[0] += rhs[0];
            m_[1] += rhs[1];
            m_[2] += rhs[2];
            m_[3] += rhs[3];
            m_[4] += rhs[4];
            m_[5] += rhs[5];
            m_[6] += rhs[6];
            m_[7] += rhs[7];
            m_[8] += rhs[8];
            m_[9] += rhs[9];
            return *this;
        }

        SymmetricMatrix operator+(const SymmetricMatrix& rhs) const
        {
            SymmetricMatrix ret = *this;
            ret += rhs;
            return ret;
        }

    private:
        double m_[10];
    };

    class MeshSimplification::Impl
    {
        struct Triangle
        {
            uint32_t indices[3];
            double err[4];
            bool deleted;
            bool dirty;
            int attr;
            glm::dvec3 normal;
        };

        struct Vertex
        {
            glm::dvec3 pos;
            uint32_t tri_start;
            uint32_t tri_count;
            SymmetricMatrix q;
            bool boundary;
        };

        struct Ref
        {
            uint32_t tri_id;
            uint32_t tri_vertex;
        };

    public:
        Mesh Process(const Mesh& input_mesh, float face_ratio)
        {
            assert(input_mesh.MeshVertexDesc().Attribs().size() == 1);

            constexpr double Aggressiveness = 7;

            vertices_.resize(input_mesh.NumVertices());
            for (uint32_t i = 0; i < input_mesh.NumVertices(); ++i)
            {
                const glm::vec3& pos = input_mesh.VertexData<glm::vec3>(i, 0);
                vertices_[i].pos = pos;
            }

            triangles_.resize(input_mesh.IndexBuffer().size() / 3);
            for (uint32_t i = 0; i < static_cast<uint32_t>(triangles_.size()); ++i)
            {
                auto& tri = triangles_[i];
                tri.indices[0] = input_mesh.Index(i * 3 + 0);
                tri.indices[1] = input_mesh.Index(i * 3 + 1);
                tri.indices[2] = input_mesh.Index(i * 3 + 2);

                tri.deleted = false;
            }

            uint32_t deleted_triangles = 0;
            std::vector<bool> deleted0;
            std::vector<bool> deleted1;
            const size_t num_triangles = triangles_.size();
            const size_t num_target_triangles = static_cast<size_t>(num_triangles * face_ratio + 0.5f);
            uint32_t iter = 0;
            while (num_triangles - deleted_triangles > num_target_triangles)
            {
                if (iter % 5 == 0)
                {
                    this->UpdateMesh(iter);
                }

                for (auto& tri : triangles_)
                {
                    tri.dirty = false;
                }

                // All triangles with edges below the threshold will be removed
                //
                // The following numbers works well for most models.
                // If it does not, try to adjust the 3 parameters
                double threshold = 0.000000001 * pow(iter + 3.0, Aggressiveness);

                for (const auto& tri : triangles_)
                {
                    if ((tri.err[3] > threshold) || tri.deleted || tri.dirty)
                    {
                        continue;
                    }

                    for (uint32_t i = 0; i < 3; ++i)
                    {
                        if (tri.err[i] < threshold)
                        {
                            const uint32_t ind0 = tri.indices[i];
                            Vertex& v0 = vertices_[ind0];

                            const uint32_t ind1 = tri.indices[(i + 1) % 3];
                            const Vertex& v1 = vertices_[ind1];

                            if (v0.boundary != v1.boundary)
                            {
                                continue;
                            }

                            // Compute vertex to collapse to
                            glm::dvec3 pos;
                            this->CalcEdgeError(ind0, ind1, pos);

                            deleted0.resize(v0.tri_count);
                            deleted1.resize(v1.tri_count);

                            // don't remove if flipped
                            if (this->Flipped(pos, ind0, ind1, v0, v1, deleted0) || this->Flipped(pos, ind1, ind0, v1, v0, deleted1))
                            {
                                continue;
                            }

                            // not flipped, so remove edge
                            v0.pos = pos;
                            v0.q = v1.q + v0.q;
                            const uint32_t tri_start = static_cast<uint32_t>(refs_.size());

                            this->UpdateTriangles(ind0, v0, deleted0, deleted_triangles);
                            this->UpdateTriangles(ind0, v1, deleted1, deleted_triangles);

                            const uint32_t tri_count = static_cast<uint32_t>(refs_.size() - tri_start);
                            if (tri_count <= v0.tri_count)
                            {
                                if (tri_count != 0)
                                {
                                    std::memcpy(&refs_[v0.tri_start], &refs_[tri_start], tri_count * sizeof(Ref));
                                }
                            }
                            else
                            {
                                // append
                                v0.tri_start = tri_start;
                            }

                            v0.tri_count = tri_count;
                            break;
                        }
                    }

                    if (num_triangles - deleted_triangles <= num_target_triangles)
                    {
                        break;
                    }
                }

                ++iter;
            }

            this->CompactMesh();

            Mesh mesh(input_mesh.MeshVertexDesc(), static_cast<uint32_t>(vertices_.size()), static_cast<uint32_t>(triangles_.size() * 3));
            for (uint32_t i = 0; i < mesh.NumVertices(); ++i)
            {
                mesh.VertexData<glm::vec3>(i, 0) = vertices_[i].pos;
            }
            for (uint32_t i = 0; i < static_cast<uint32_t>(triangles_.size()); ++i)
            {
                const auto& tri = triangles_[i];
                mesh.Index(i * 3 + 0) = tri.indices[0];
                mesh.Index(i * 3 + 1) = tri.indices[1];
                mesh.Index(i * 3 + 2) = tri.indices[2];
            }

            return mesh;
        }

    private:
        // Compact triangles, compute edge error and build reference list
        void UpdateMesh(uint32_t iter)
        {
            if (iter > 0)
            {
                uint32_t dst = 0;
                for (size_t i = 0; i < triangles_.size(); ++i)
                {
                    if (!triangles_[i].deleted)
                    {
                        triangles_[dst] = std::move(triangles_[i]);
                        ++dst;
                    }
                }
                triangles_.resize(dst);
            }

            for (auto& vert : vertices_)
            {
                vert.tri_start = 0;
                vert.tri_count = 0;
            }
            for (const auto& tri : triangles_)
            {
                for (const uint32_t ind : tri.indices)
                {
                    ++vertices_[ind].tri_count;
                }
            }
            uint32_t tri_start = 0;
            for (auto& vert : vertices_)
            {
                vert.tri_start = tri_start;
                tri_start += vert.tri_count;
                vert.tri_count = 0;
            }

            refs_.resize(triangles_.size() * 3);
            for (size_t i = 0; i < triangles_.size(); ++i)
            {
                const Triangle& tri = triangles_[i];
                for (uint32_t j = 0; j < 3; ++j)
                {
                    Vertex& vert = vertices_[tri.indices[j]];
                    refs_[vert.tri_start + vert.tri_count].tri_id = static_cast<uint32_t>(i);
                    refs_[vert.tri_start + vert.tri_count].tri_vertex = j;
                    ++vert.tri_count;
                }
            }

            // Init Quadrics by Plane & Edge Errors
            // required at the beginning (iter == 0) recomputing during the simplification is not required, but mostly improves the
            // result for closed meshes
            if (iter == 0)
            {
                for (auto& vert : vertices_)
                {
                    vert.boundary = false;
                }

                std::vector<uint32_t> vert_count;
                std::vector<uint32_t> vert_ids;
                for (const auto& vert : vertices_)
                {
                    vert_count.clear();
                    vert_ids.clear();
                    for (uint32_t i = 0; i < vert.tri_count; ++i)
                    {
                        const auto& tri = triangles_[refs_[vert.tri_start + i].tri_id];
                        for (uint32_t k = 0; k < 3; ++k)
                        {
                            uint32_t ofs = 0;
                            const uint32_t id = tri.indices[k];
                            while (ofs < vert_count.size())
                            {
                                if (vert_ids[ofs] == id)
                                {
                                    break;
                                }
                                ++ofs;
                            }
                            if (ofs == vert_count.size())
                            {
                                vert_count.push_back(1);
                                vert_ids.push_back(id);
                            }
                            else
                            {
                                ++vert_count[ofs];
                            }
                        }
                    }
                    for (uint32_t i = 0; i < vert_count.size(); ++i)
                    {
                        if (vert_count[i] == 1)
                        {
                            vertices_[vert_ids[i]].boundary = true;
                        }
                    }
                }

                for (auto& vert : vertices_)
                {
                    vert.q = SymmetricMatrix();
                }

                for (auto& tri : triangles_)
                {
                    glm::dvec3 pos[3];
                    for (uint32_t i = 0; i < 3; ++i)
                    {
                        pos[i] = vertices_[tri.indices[i]].pos;
                    }

                    tri.normal = glm::normalize(glm::cross(pos[1] - pos[0], pos[2] - pos[0]));

                    for (const uint32_t ind : tri.indices)
                    {
                        vertices_[ind].q += SymmetricMatrix(tri.normal.x, tri.normal.y, tri.normal.z, -glm::dot(tri.normal, pos[0]));
                    }
                }
                for (auto& tri : triangles_)
                {
                    glm::dvec3 pos;
                    for (uint32_t i = 0; i < 3; ++i)
                    {
                        tri.err[i] = this->CalcEdgeError(tri.indices[i], tri.indices[(i + 1) % 3], pos);
                    }
                    tri.err[3] = std::min({tri.err[0], tri.err[1], tri.err[2]});
                }
            }
        }

        // Error between vertex and Quadric
        double CalcVertexError(const SymmetricMatrix& q, const glm::dvec3& pos) const
        {
            return q[0] * pos.x * pos.x + 2 * q[1] * pos.x * pos.y + 2 * q[2] * pos.x * pos.z + 2 * q[3] * pos.x + q[4] * pos.y * pos.y +
                   2 * q[5] * pos.y * pos.z + 2 * q[6] * pos.y + q[7] * pos.z * pos.z + 2 * q[8] * pos.z + q[9];
        }

        // Error for one edge
        double CalcEdgeError(uint32_t v0, uint32_t v1, glm::dvec3& pos_result) const
        {
            // compute interpolated vertex

            const SymmetricMatrix q = vertices_[v0].q + vertices_[v1].q;
            const bool boundary = vertices_[v0].boundary && vertices_[v1].boundary;
            double error = 0;
            double det = q.Det(0, 1, 2, 1, 4, 5, 2, 5, 7);
            if ((det != 0) && !boundary)
            {
                // q_delta is invertible
                pos_result.x = -1 / det * (q.Det(1, 2, 3, 4, 5, 6, 5, 7, 8)); // vx = A41/det(q_delta)
                pos_result.y = 1 / det * (q.Det(0, 2, 3, 1, 5, 6, 2, 7, 8));  // vy = A42/det(q_delta)
                pos_result.z = -1 / det * (q.Det(0, 1, 3, 1, 4, 6, 2, 5, 8)); // vz = A43/det(q_delta)

                error = this->CalcVertexError(q, pos_result);
            }
            else
            {
                // det = 0 -> try to find best result
                const glm::dvec3& p1 = vertices_[v0].pos;
                const glm::dvec3& p2 = vertices_[v1].pos;
                const glm::dvec3 p3 = (p1 + p2) / 2.0;
                error = this->CalcVertexError(q, p1);
                pos_result = p1;
                const double error2 = this->CalcVertexError(q, p2);
                if (error2 < error)
                {
                    error = error2;
                    pos_result = p2;
                }
                const double error3 = this->CalcVertexError(q, p3);
                if (error3 < error)
                {
                    error = error3;
                    pos_result = p3;
                }
            }

            return error;
        }

        // Check if a triangle flips when this edge is removed
        bool Flipped(const glm::dvec3& pos, [[maybe_unused]] uint32_t v0, uint32_t v1, const Vertex& vert0,
            [[maybe_unused]] const Vertex& vert1, std::vector<bool>& deleted)
        {
            for (uint32_t i = 0; i < vert0.tri_count; ++i)
            {
                const auto& ref = refs_[vert0.tri_start + i];
                const auto& tri = triangles_[ref.tri_id];
                if (tri.deleted)
                {
                    continue;
                }

                uint32_t s = ref.tri_vertex;
                uint32_t id0 = tri.indices[(s + 1) % 3];
                uint32_t id1 = tri.indices[(s + 2) % 3];

                if ((id0 == v1) || (id1 == v1)) // delete ?
                {
                    deleted[i] = true;
                    continue;
                }

                glm::dvec3 d0 = glm::normalize(vertices_[id0].pos - pos);
                glm::dvec3 d1 = glm::normalize(vertices_[id1].pos - pos);
                if (std::abs(glm::dot(d0, d1)) > 0.999)
                {
                    return true;
                }

                glm::dvec3 normal = glm::normalize(glm::cross(d0, d1));
                deleted[i] = false;
                if (glm::dot(normal, tri.normal) < 0.2)
                {
                    return true;
                }
            }

            return false;
        }

        // Update triangle connections and edge error after a edge is collapsed
        void UpdateTriangles(uint32_t i0, const Vertex& vert, const std::vector<bool>& deleted, uint32_t& deleted_triangles)
        {
            glm::dvec3 pos;
            for (uint32_t i = 0; i < vert.tri_count; ++i)
            {
                const auto& ref = refs_[vert.tri_start + i];
                auto& tri = triangles_[ref.tri_id];
                if (tri.deleted)
                {
                    continue;
                }

                if (deleted[i])
                {
                    tri.deleted = true;
                    ++deleted_triangles;
                    continue;
                }

                tri.indices[ref.tri_vertex] = i0;
                tri.dirty = true;
                tri.err[0] = this->CalcEdgeError(tri.indices[0], tri.indices[1], pos);
                tri.err[1] = this->CalcEdgeError(tri.indices[1], tri.indices[2], pos);
                tri.err[2] = this->CalcEdgeError(tri.indices[2], tri.indices[0], pos);
                tri.err[3] = std::min({tri.err[0], tri.err[1], tri.err[2]});
                refs_.push_back(ref);
            }
        }

        // Finally compact mesh before exiting
        void CompactMesh()
        {
            size_t dst = 0;
            for (auto& vert : vertices_)
            {
                vert.tri_count = 0;
            }
            for (const auto& tri : triangles_)
            {
                if (!tri.deleted)
                {
                    triangles_[dst] = tri;
                    ++dst;
                    for (size_t i = 0; i < 3; ++i)
                    {
                        vertices_[tri.indices[i]].tri_count = 1;
                    }
                }
            }
            triangles_.resize(dst);

            dst = 0;
            for (auto& vert : vertices_)
            {
                if (vert.tri_count > 0)
                {
                    vert.tri_start = static_cast<uint32_t>(dst);
                    vertices_[dst].pos = vert.pos;
                    ++dst;
                }
            }
            for (auto& tri : triangles_)
            {
                for (size_t i = 0; i < 3; ++i)
                {
                    tri.indices[i] = vertices_[tri.indices[i]].tri_start;
                }
            }
            vertices_.resize(dst);
        }

    private:
        std::vector<Triangle> triangles_;
        std::vector<Vertex> vertices_;
        std::vector<Ref> refs_;
    };

    MeshSimplification::MeshSimplification() : impl_(std::make_unique<Impl>())
    {
    }

    MeshSimplification::~MeshSimplification() noexcept = default;

    MeshSimplification::MeshSimplification(MeshSimplification&& other) noexcept = default;
    MeshSimplification& MeshSimplification::operator=(MeshSimplification&& other) noexcept = default;

    Mesh MeshSimplification::Process(const Mesh& input_mesh, float face_ratio)
    {
        return impl_->Process(input_mesh, face_ratio);
    }
} // namespace AIHoloImager
