// Copyright (c) 2024 Minmin Gong
//

#include "MeshGenerator.hpp"

#include <array>
#include <cassert>
#include <set>

#include <directx/d3d12.h>
#include <xatlas/xatlas.h>

#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuResourceViews.hpp"
#include "Gpu/GpuTexture.hpp"

#include "CompiledShader/FillPosTexturePs.h"
#include "CompiledShader/FillPosTextureVs.h"
#include "CompiledShader/GetPosListCs.h"

using namespace DirectX;

namespace AIHoloImager
{
    class MeshGenerator::Impl
    {
    public:
        Impl(GpuSystem& gpu_system, PythonSystem& python_system) : gpu_system_(gpu_system), python_system_(python_system)
        {
            mesh_generator_module_ = python_system_.Import("MeshGenerator");
            mesh_generator_class_ = python_system_.GetAttr(*mesh_generator_module_, "MeshGenerator");
            mesh_generator_ = python_system_.CallObject(*mesh_generator_class_);
            mesh_generator_gen_pos_mesh_method_ = python_system_.GetAttr(*mesh_generator_, "GenPosMesh");
            mesh_generator_query_colors_method_ = python_system_.GetAttr(*mesh_generator_, "QueryColors");

            pil_module_ = python_system_.Import("PIL");
            image_class_ = python_system_.GetAttr(*pil_module_, "Image");
            image_frombuffer_method_ = python_system_.GetAttr(*image_class_, "frombuffer");

            {
                const ShaderInfo shaders[] = {
                    {FillPosTextureVs_shader, 0, 0, 0},
                    {FillPosTexturePs_shader, 0, 0, 0},
                };

                const DXGI_FORMAT rtv_formats[] = {DXGI_FORMAT_R32G32B32A32_FLOAT};

                GpuRenderPipeline::States states;
                states.cull_mode = GpuRenderPipeline::CullMode::None;
                states.conservative_raster = true;
                states.depth_enable = false;
                states.rtv_formats = rtv_formats;
                states.dsv_format = DXGI_FORMAT_UNKNOWN;

                const D3D12_INPUT_ELEMENT_DESC input_elems[] = {
                    {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
                    {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
                };

                fill_pos_texture_pipeline_ = GpuRenderPipeline(gpu_system_, shaders, input_elems, {}, states);
            }
            {
                const ShaderInfo shader = {GetPosListCs_shader, 0, 1, 3};
                get_pos_list_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }
        }

        Mesh Generate(std::span<const Texture> input_images, uint32_t texture_size, [[maybe_unused]] const std::filesystem::path& tmp_dir)
        {
            assert(input_images.size() == 6);
            assert(input_images[0].Width() == 320);
            assert(input_images[0].Height() == 320);
            assert(input_images[0].NumChannels() == 3);

#ifdef AIHI_KEEP_INTERMEDIATES
            auto output_dir = tmp_dir / "Mesh";
            std::filesystem::create_directories(output_dir);
#endif

            PyObjectPtr py_input_images[6];
            for (size_t i = 0; i < input_images.size(); ++i)
            {
                auto& input_image = input_images[i];

                auto args = python_system_.MakeTuple(3);
                {
                    python_system_.SetTupleItem(*args, 0, python_system_.MakeObject(L"RGB"));
                }
                {
                    auto size = python_system_.MakeTuple(2);
                    {
                        python_system_.SetTupleItem(*size, 0, python_system_.MakeObject(input_image.Width()));
                        python_system_.SetTupleItem(*size, 1, python_system_.MakeObject(input_image.Height()));
                    }
                    python_system_.SetTupleItem(*args, 1, std::move(size));
                }
                {
                    auto image = python_system_.MakeObject(
                        std::span<const std::byte>(reinterpret_cast<const std::byte*>(input_image.Data()), input_image.DataSize()));
                    python_system_.SetTupleItem(*args, 2, std::move(image));
                }

                py_input_images[i] = python_system_.CallObject(*image_frombuffer_method_, *args);
            }

            auto args = python_system_.MakeTuple(1);
            {
                auto imgs_args = python_system_.MakeTuple(std::size(py_input_images));
                for (uint32_t i = 0; i < std::size(py_input_images); ++i)
                {
                    python_system_.SetTupleItem(*imgs_args, i, std::move(py_input_images[i]));
                }
                python_system_.SetTupleItem(*args, 0, std::move(imgs_args));
            }

            auto verts_faces = python_system_.CallObject(*mesh_generator_gen_pos_mesh_method_, *args);

            Mesh pos_only_mesh;
            {
                auto verts = python_system_.GetTupleItem(*verts_faces, 0);
                auto faces = python_system_.GetTupleItem(*verts_faces, 1);

                auto verts_tobytes_method = python_system_.GetAttr(*verts, "tobytes");
                auto verts_data = python_system_.CallObject(*verts_tobytes_method);
                const auto positions = python_system_.ToSpan<const XMFLOAT3>(*verts_data);

                auto faces_tobytes_method = python_system_.GetAttr(*faces, "tobytes");
                auto faces_data = python_system_.CallObject(*faces_tobytes_method);
                const auto indices = python_system_.ToSpan<const uint32_t>(*faces_data);

                pos_only_mesh = CleanMesh(positions, indices);

#ifdef AIHI_KEEP_INTERMEDIATES
                SaveMesh(pos_only_mesh, output_dir / "AiMeshPosOnly.glb");
#endif
            }

            Mesh pos_uv_mesh = UnwrapUv(pos_only_mesh, texture_size);

            GpuReadbackBuffer counter_cpu_buff;
            GpuReadbackBuffer uv_cpu_buff;
            GpuReadbackBuffer pos_cpu_buff;
            {
                auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

                GpuTexture2D pos_tex = GenPosTex(cmd_list, pos_uv_mesh, texture_size);

                GpuBuffer counter_buff;
                GpuBuffer uv_buff;
                GpuBuffer pos_buff;
                PosTexToList(cmd_list, pos_tex, counter_buff, uv_buff, pos_buff);

                counter_cpu_buff = GpuReadbackBuffer(gpu_system_, counter_buff.Size(), L"counter_cpu_buff");
                cmd_list.Copy(counter_cpu_buff, counter_buff);

                uv_cpu_buff = GpuReadbackBuffer(gpu_system_, uv_buff.Size(), L"uv_cpu_buff");
                cmd_list.Copy(uv_cpu_buff, uv_buff);

                pos_cpu_buff = GpuReadbackBuffer(gpu_system_, pos_buff.Size(), L"pos_cpu_buff");
                cmd_list.Copy(pos_cpu_buff, pos_buff);

                gpu_system_.Execute(std::move(cmd_list));
                gpu_system_.WaitForGpu();
            }

            Texture ai_tex = this->GenTexture(counter_cpu_buff, uv_cpu_buff, pos_cpu_buff, texture_size);
            pos_uv_mesh.AlbedoTexture(std::move(ai_tex));

#ifdef AIHI_KEEP_INTERMEDIATES
            SaveMesh(pos_uv_mesh, output_dir / "AiMesh.glb");
#endif
            return pos_uv_mesh;
        }

    private:
        Mesh CleanMesh(std::span<const XMFLOAT3> positions, std::span<const uint32_t> indices)
        {
            constexpr float Scale = 1e5f;

            std::set<std::array<int32_t, 3>> unique_int_pos;
            for (uint32_t i = 0; i < positions.size(); ++i)
            {
                const XMFLOAT3& pos = positions[i];
                std::array<int32_t, 3> int_pos = {static_cast<int32_t>(pos.x * Scale + 0.5f), static_cast<int32_t>(pos.y * Scale + 0.5f),
                    static_cast<int32_t>(pos.z * Scale + 0.5f)};
                unique_int_pos.emplace(std::move(int_pos));
            }

            Mesh ret_mesh(static_cast<uint32_t>(unique_int_pos.size()), static_cast<uint32_t>(indices.size()));

            std::vector<std::array<int32_t, 3>> unique_int_pos_vec(unique_int_pos.begin(), unique_int_pos.end());
            std::vector<uint32_t> vertex_mapping(positions.size());
#ifdef _OPENMP
    #pragma omp parallel
#endif
            for (uint32_t i = 0; i < positions.size(); ++i)
            {
                const XMFLOAT3& pos = positions[i];
                const std::array<int32_t, 3> int_pos = {static_cast<int32_t>(pos.x * Scale + 0.5f),
                    static_cast<int32_t>(pos.y * Scale + 0.5f), static_cast<int32_t>(pos.z * Scale + 0.5f)};

                const auto iter = std::lower_bound(unique_int_pos_vec.begin(), unique_int_pos_vec.end(), int_pos);
                assert(*iter == int_pos);

                vertex_mapping[i] = static_cast<uint32_t>(iter - unique_int_pos_vec.begin());

                auto& pos_only_vert = ret_mesh.Vertex(vertex_mapping[i]);
                pos_only_vert.pos = pos;
                pos_only_vert.texcoord = XMFLOAT2(0, 0);
            }

            uint32_t num_faces = 0;
            for (size_t i = 0; i < indices.size(); i += 3)
            {
                uint32_t face[3];
                for (uint32_t j = 0; j < 3; ++j)
                {
                    face[j] = vertex_mapping[indices[i + j]];
                }

                bool degenerated = false;
                for (uint32_t j = 0; j < 3; ++j)
                {
                    if (face[j] == face[(j + 1) % 3])
                    {
                        degenerated = true;
                        break;
                    }
                }

                if (!degenerated)
                {
                    for (uint32_t j = 0; j < 3; ++j)
                    {
                        ret_mesh.Index(num_faces * 3 + j) = face[j];
                    }
                    ++num_faces;
                }
            }

            ret_mesh.ResizeIndices(num_faces * 3);

            return ret_mesh;
        }

        Mesh UnwrapUv(const Mesh& input_mesh, uint32_t texture_size)
        {
            Mesh ret_mesh;

            xatlas::Atlas* atlas = xatlas::Create();

            xatlas::MeshDecl mesh_decl;
            mesh_decl.vertexCount = static_cast<uint32_t>(input_mesh.Vertices().size());
            mesh_decl.vertexPositionData = input_mesh.Vertices().data();
            mesh_decl.vertexPositionStride = sizeof(input_mesh.Vertices()[0]);
            mesh_decl.indexCount = static_cast<uint32_t>(input_mesh.Indices().size());
            mesh_decl.indexData = input_mesh.Indices().data();
            mesh_decl.indexFormat = xatlas::IndexFormat::UInt32;

            xatlas::AddMeshError error = xatlas::AddMesh(atlas, mesh_decl, 1);
            if (error == xatlas::AddMeshError::Success)
            {
                xatlas::ChartOptions chart_options;

                xatlas::PackOptions pack_options;
                pack_options.padding = 2;
                pack_options.texelsPerUnit = 0;
                pack_options.resolution = texture_size;

                xatlas::Generate(atlas, chart_options, pack_options);

                ret_mesh = Mesh(0, 0);
                for (uint32_t mi = 0; mi < atlas->meshCount; ++mi)
                {
                    const uint32_t base_vertex = static_cast<uint32_t>(ret_mesh.Vertices().size());

                    const xatlas::Mesh& mesh = atlas->meshes[mi];
                    ret_mesh.ResizeVertices(static_cast<uint32_t>(ret_mesh.Vertices().size() + mesh.vertexCount));
                    for (uint32_t vi = 0; vi < mesh.vertexCount; ++vi)
                    {
                        const auto& vertex = mesh.vertexArray[vi];
                        const auto& pos = input_mesh.Vertex(vertex.xref).pos;
                        const XMFLOAT2 uv(vertex.uv[0] / atlas->width, vertex.uv[1] / atlas->height);
                        ret_mesh.Vertex(base_vertex + vi) = {pos, uv};
                    }

                    ret_mesh.ResizeIndices(static_cast<uint32_t>(ret_mesh.Indices().size() + mesh.indexCount));
                    for (uint32_t i = 0; i < mesh.indexCount; ++i)
                    {
                        ret_mesh.Index(i) = base_vertex + mesh.indexArray[i];
                    }
                }
            }

            xatlas::Destroy(atlas);

            if (error != xatlas::AddMeshError::Success)
            {
                throw std::runtime_error(std::format("UV unwrapping failed {}", static_cast<uint32_t>(error)));
            }

            return ret_mesh;
        }

        GpuTexture2D GenPosTex(GpuCommandList& cmd_list, const Mesh& mesh, uint32_t texture_size)
        {
            GpuBuffer vb(gpu_system_, static_cast<uint32_t>(mesh.Vertices().size() * sizeof(Mesh::VertexFormat)), D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, L"vb");
            memcpy(vb.Map(), mesh.Vertices().data(), vb.Size());
            vb.Unmap(D3D12_RANGE{0, vb.Size()});

            GpuBuffer ib(gpu_system_, static_cast<uint32_t>(mesh.Indices().size() * sizeof(uint32_t)), D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, L"ib");
            memcpy(ib.Map(), mesh.Indices().data(), ib.Size());
            ib.Unmap(D3D12_RANGE{0, ib.Size()});

            auto* d3d12_cmd_list = cmd_list.NativeCommandList<ID3D12GraphicsCommandList>();

            GpuTexture2D pos_tex(gpu_system_, texture_size, texture_size, 1, DXGI_FORMAT_R32G32B32A32_FLOAT,
                D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_COMMON, L"pos_tex");
            GpuRenderTargetView pos_tex_rtv(gpu_system_, pos_tex);

            pos_tex.Transition(cmd_list, D3D12_RESOURCE_STATE_RENDER_TARGET);

            const float clear_clr[] = {0, 0, 0, 0};
            d3d12_cmd_list->ClearRenderTargetView(pos_tex_rtv.CpuHandle(), clear_clr, 0, nullptr);

            const GpuCommandList::VertexBufferBinding vb_bindings[] = {{&vb, 0, sizeof(Mesh::VertexFormat)}};
            const GpuCommandList::IndexBufferBinding ib_binding = {&ib, 0, DXGI_FORMAT_R32_UINT};

            const GpuCommandList::ShaderBinding shader_bindings[] = {
                {{}, {}, {}},
                {{}, {}, {}},
            };

            const GpuRenderTargetView* rtvs[] = {&pos_tex_rtv};

            const D3D12_VIEWPORT viewports[] = {{0, 0, static_cast<float>(texture_size), static_cast<float>(texture_size), 0, 1}};
            const D3D12_RECT scissor_rcs[] = {{0, 0, static_cast<LONG>(texture_size), static_cast<LONG>(texture_size)}};

            const uint32_t num_indices = static_cast<uint32_t>(mesh.Indices().size());
            cmd_list.Render(
                fill_pos_texture_pipeline_, vb_bindings, &ib_binding, num_indices, shader_bindings, rtvs, nullptr, viewports, scissor_rcs);

            pos_tex.Transition(cmd_list, D3D12_RESOURCE_STATE_COMMON);

            return pos_tex;
        }

        void PosTexToList(
            GpuCommandList& cmd_list, const GpuTexture2D& pos_tex, GpuBuffer& counter_buff, GpuBuffer& uv_buff, GpuBuffer& pos_buff)
        {
            constexpr uint32_t BlockDim = 16;

            GpuShaderResourceView pos_tex_srv(gpu_system_, pos_tex);

            counter_buff = GpuBuffer(gpu_system_, sizeof(uint32_t), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON, L"counter_buff");
            GpuUnorderedAccessView counter_uav(gpu_system_, counter_buff, sizeof(uint32_t));

            const uint32_t max_pos_size = pos_tex.Width(0) * pos_tex.Height(0);
            uv_buff = GpuBuffer(gpu_system_, max_pos_size * sizeof(XMUINT2), D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON, L"uv_buff");
            GpuUnorderedAccessView uv_uav(gpu_system_, uv_buff, sizeof(XMUINT2));
            pos_buff = GpuBuffer(gpu_system_, max_pos_size * sizeof(XMFLOAT3), D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON, L"pos_buff");
            GpuUnorderedAccessView pos_uav(gpu_system_, pos_buff, sizeof(float));

            const GpuShaderResourceView* srvs[] = {&pos_tex_srv};
            GpuUnorderedAccessView* uavs[] = {&counter_uav, &uv_uav, &pos_uav};
            const GpuCommandList::ShaderBinding shader_binding = {{}, srvs, uavs};
            cmd_list.Compute(
                get_pos_list_pipeline_, DivUp(pos_tex.Width(0), BlockDim), DivUp(pos_tex.Height(0), BlockDim), 1, shader_binding);
        }

        Texture GenTexture(const GpuReadbackBuffer& counter_cpu_buff, const GpuReadbackBuffer& uv_cpu_buff,
            const GpuReadbackBuffer& pos_cpu_buff, uint32_t texture_size)
        {
            const uint32_t count = *counter_cpu_buff.MappedData<uint32_t>();

            const XMUINT2* uv = uv_cpu_buff.MappedData<XMUINT2>();
            const XMFLOAT3* pos = pos_cpu_buff.MappedData<XMFLOAT3>();

            auto query_colors_args = python_system_.MakeTuple(2);
            {
                auto pos_py = python_system_.MakeObject(
                    std::span<const std::byte>(reinterpret_cast<const std::byte*>(pos), count * sizeof(XMFLOAT3)));
                python_system_.SetTupleItem(*query_colors_args, 0, std::move(pos_py));
                python_system_.SetTupleItem(*query_colors_args, 1, python_system_.MakeObject(count));
            }

            auto colors_py = python_system_.CallObject(*mesh_generator_query_colors_method_, *query_colors_args);

            auto colors_tobytes_method = python_system_.GetAttr(*colors_py, "tobytes");
            auto colors_data = python_system_.CallObject(*colors_tobytes_method);
            const auto colors = python_system_.ToSpan<const uint32_t>(*colors_data);

            Texture tex(texture_size, texture_size, 4);
            uint32_t* tex_data = reinterpret_cast<uint32_t*>(tex.Data());
            std::memset(tex_data, 0, tex.DataSize());
#ifdef _OPENMP
    #pragma omp parallel
#endif
            for (uint32_t i = 0; i < count; ++i)
            {
                tex_data[uv[i].y * texture_size + uv[i].x] = colors[i];
            }

            return tex;
        }

    private:
        GpuSystem& gpu_system_;
        PythonSystem& python_system_;

        PyObjectPtr mesh_generator_module_;
        PyObjectPtr mesh_generator_class_;
        PyObjectPtr mesh_generator_;
        PyObjectPtr mesh_generator_gen_pos_mesh_method_;
        PyObjectPtr mesh_generator_query_colors_method_;

        PyObjectPtr pil_module_;
        PyObjectPtr image_class_;
        PyObjectPtr image_frombuffer_method_;

        GpuRenderPipeline fill_pos_texture_pipeline_;
        GpuComputePipeline get_pos_list_pipeline_;
    };

    MeshGenerator::MeshGenerator(GpuSystem& gpu_system, PythonSystem& python_system)
        : impl_(std::make_unique<Impl>(gpu_system, python_system))
    {
    }

    MeshGenerator::~MeshGenerator() noexcept = default;

    MeshGenerator::MeshGenerator(MeshGenerator&& other) noexcept = default;
    MeshGenerator& MeshGenerator::operator=(MeshGenerator&& other) noexcept = default;

    Mesh MeshGenerator::Generate(std::span<const Texture> input_images, uint32_t texture_size, const std::filesystem::path& tmp_dir)
    {
        return impl_->Generate(std::move(input_images), texture_size, tmp_dir);
    }
} // namespace AIHoloImager
