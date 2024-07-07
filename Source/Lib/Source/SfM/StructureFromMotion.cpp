// Copyright (c) 2024 Minmin Gong
//

#include "StructureFromMotion.hpp"

#include <algorithm>
#include <format>
#include <map>
#include <stdexcept>

#include <DirectXMath.h>

#ifndef OPENMVG_USE_OPENMP
    #define OPENMVG_USE_OPENMP
#endif

#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4018) // Ignore signed/unsigned compare mismatch
    #pragma warning(disable : 4100) // Ignore unreferenced formal parameter
    #pragma warning(disable : 4127) // Ignore conditional expression is constant
    #pragma warning(disable : 4244) // Ignore implicit conversion
    #pragma warning(disable : 4267) // Ignore implicit conversion
    #pragma warning(disable : 4305) // Ignore transcation from double to float
    #pragma warning(disable : 4702) // Ignore unreachable code
    #pragma warning(disable : 5054) // Ignore operator between enums of different types
    #pragma warning(disable : 5055) // Ignore operator between enums and floating-point types
#endif
#include <openMVG/cameras/Camera_Pinhole_Radial.hpp>
#include <openMVG/cameras/Camera_undistort_image.hpp>
#include <openMVG/exif/exif_IO_EasyExif.hpp>
#include <openMVG/exif/sensor_width_database/ParseDatabase.hpp>
#include <openMVG/features/regions.hpp>
#include <openMVG/features/sift/SIFT_Anatomy_Image_Describer.hpp>
#include <openMVG/image/image_io.hpp>
#include <openMVG/matching_image_collection/Cascade_Hashing_Matcher_Regions.hpp>
#include <openMVG/matching_image_collection/E_ACRobust.hpp>
#include <openMVG/matching_image_collection/F_ACRobust.hpp>
#include <openMVG/matching_image_collection/GeometricFilter.hpp>
#include <openMVG/matching_image_collection/Pair_Builder.hpp>
#include <openMVG/sfm/pipelines/global/GlobalSfM_rotation_averaging.hpp>
#include <openMVG/sfm/pipelines/global/GlobalSfM_translation_averaging.hpp>
#include <openMVG/sfm/pipelines/global/sfm_global_engine_relative_motions.hpp>
#include <openMVG/sfm/pipelines/sequential/sequential_SfM.hpp>
#include <openMVG/sfm/pipelines/sfm_features_provider.hpp>
#include <openMVG/sfm/pipelines/sfm_matches_provider.hpp>
#include <openMVG/sfm/pipelines/sfm_regions_provider.hpp>
#include <openMVG/sfm/sfm_data.hpp>
#include <openMVG/sfm/sfm_data_io.hpp>
#include <openMVG/sfm/sfm_data_utils.hpp>
#ifdef _MSC_VER
    #pragma warning(pop)
#endif

#include "Gpu/GpuBufferHelper.hpp"
#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuResourceViews.hpp"
#include "Gpu/GpuTexture.hpp"

#include "CompiledShader/UndistortCs.h"

using namespace openMVG;
using namespace openMVG::cameras;
using namespace openMVG::sfm;
using namespace openMVG::image;
using namespace openMVG::matching;
using namespace openMVG::matching_image_collection;

using namespace DirectX;

namespace AIHoloImager
{
    class StructureFromMotion::Impl
    {
    private:
        struct FeatureRegions
        {
            std::unique_ptr<features::Regions> regions_type;
            std::vector<std::unique_ptr<features::Regions>> feature_regions;
        };

    public:
        explicit Impl(const std::filesystem::path& exe_dir, GpuSystem& gpu_system) : exe_dir_(exe_dir), gpu_system_(gpu_system)
        {
            ID3D12Device* d3d12_device = gpu_system_.NativeDevice();

            {
                undistort_cb_ = ConstantBuffer<UndistortConstantBuffer>(gpu_system_, 1, L"undistort_cb_");

                const D3D12_DESCRIPTOR_RANGE ranges[] = {
                    {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND},
                    {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND},
                };

                const D3D12_ROOT_PARAMETER root_params[] = {
                    CreateRootParameterAsConstantBufferView(0),
                    CreateRootParameterAsDescriptorTable(&ranges[0], 1),
                    CreateRootParameterAsDescriptorTable(&ranges[1], 1),
                };

                D3D12_STATIC_SAMPLER_DESC bilinear_sampler_desc{};
                bilinear_sampler_desc.Filter = D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT;
                bilinear_sampler_desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
                bilinear_sampler_desc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
                bilinear_sampler_desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
                bilinear_sampler_desc.MaxAnisotropy = 16;
                bilinear_sampler_desc.ComparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
                bilinear_sampler_desc.MinLOD = 0.0f;
                bilinear_sampler_desc.MaxLOD = D3D12_FLOAT32_MAX;
                bilinear_sampler_desc.ShaderRegister = 0;

                const D3D12_ROOT_SIGNATURE_DESC root_signature_desc = {
                    static_cast<uint32_t>(std::size(root_params)), root_params, 1, &bilinear_sampler_desc, D3D12_ROOT_SIGNATURE_FLAG_NONE};

                ComPtr<ID3DBlob> blob;
                ComPtr<ID3DBlob> error;
                HRESULT hr = ::D3D12SerializeRootSignature(&root_signature_desc, D3D_ROOT_SIGNATURE_VERSION_1, blob.Put(), error.Put());
                if (FAILED(hr))
                {
                    ::OutputDebugStringW(
                        std::format(L"D3D12SerializeRootSignature failed: {}\n", static_cast<const wchar_t*>(error->GetBufferPointer()))
                            .c_str());
                    TIFHR(hr);
                }

                TIFHR(d3d12_device->CreateRootSignature(
                    1, blob->GetBufferPointer(), blob->GetBufferSize(), UuidOf<ID3D12RootSignature>(), undistort_root_sig_.PutVoid()));

                D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc;
                pso_desc.pRootSignature = undistort_root_sig_.Get();
                pso_desc.CS.pShaderBytecode = UndistortCs_shader;
                pso_desc.CS.BytecodeLength = sizeof(UndistortCs_shader);
                pso_desc.NodeMask = 0;
                pso_desc.CachedPSO.pCachedBlob = nullptr;
                pso_desc.CachedPSO.CachedBlobSizeInBytes = 0;
                pso_desc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
                TIFHR(d3d12_device->CreateComputePipelineState(&pso_desc, UuidOf<ID3D12PipelineState>(), undistort_pso_.PutVoid()));
            }
        }

        ~Impl() noexcept
        {
            undistort_cb_ = ConstantBuffer<UndistortConstantBuffer>();
            undistort_root_sig_.Reset();
            undistort_pso_.Reset();

            gpu_system_.WaitForGpu();
        }

        Result Process(const std::filesystem::path& input_path, bool sequential, const std::filesystem::path& tmp_dir)
        {
            SfM_Data sfm_data = this->IntrinsicAnalysis(input_path);
            FeatureRegions regions = this->FeatureExtraction(sfm_data);
            const PairWiseMatches map_putative_matches = this->PairMatching(sfm_data, regions);
            const PairWiseMatches map_geometric_matches = this->GeometricFilter(sfm_data, map_putative_matches, regions, sequential);

            const auto sfm_tmp_dir = tmp_dir / "Sfm";
            std::filesystem::create_directories(sfm_tmp_dir);

            const SfM_Data processed_sfm_data =
                this->PointCloudReconstruction(sfm_data, map_geometric_matches, regions, sequential, sfm_tmp_dir);
            return ExportResult(processed_sfm_data);
        }

    private:
        SfM_Data IntrinsicAnalysis(const std::filesystem::path& image_dir) const
        {
            // Reference from openMVG/src/software/SfM/main_SfMInit_ImageListing.cpp

            const auto camera_sensor_db_path = exe_dir_ / "sensor_width_camera_database.txt";

            std::vector<Datasheet> vec_database;
            if (!parseDatabase(camera_sensor_db_path.string(), vec_database))
            {
                throw std::runtime_error(
                    std::format("Invalid input database: {}, please specify a valid file.", camera_sensor_db_path.string()));
            }

            std::vector<std::filesystem::path> input_image_paths;
            for (const auto& dir_entry : std::filesystem::directory_iterator{image_dir})
            {
                if (dir_entry.is_regular_file())
                {
                    input_image_paths.push_back(std::filesystem::relative(dir_entry.path(), image_dir).string());
                }
            }
            std::sort(input_image_paths.begin(), input_image_paths.end());

            SfM_Data sfm_data;
            sfm_data.s_root_path = image_dir.string();

            Views& views = sfm_data.views;
            Intrinsics& intrinsics = sfm_data.intrinsics;

            exif::Exif_IO_EasyExif exif_reader;
            for (auto iter = input_image_paths.begin(); iter != input_image_paths.end(); ++iter)
            {
                const std::filesystem::path image_file_path = image_dir / *iter;

                if (GetFormat(image_file_path.string().c_str()) == Unknown)
                {
                    std::cerr << image_file_path.filename() << ": Unsupported image file format.\n";
                    continue;
                }

                ImageHeader img_header;
                if (!ReadImageHeader(image_file_path.string().c_str(), &img_header))
                {
                    continue;
                }

                const uint32_t width = static_cast<uint32_t>(img_header.width);
                const uint32_t height = static_cast<uint32_t>(img_header.height);
                const double ppx = width / 2.0;
                const double ppy = height / 2.0;
                double focal = -1;

                exif_reader.open(image_file_path.string());
                if (exif_reader.doesHaveExifInfo())
                {
                    if (exif_reader.getFocal() == 0.0f)
                    {
                        std::cerr << image_file_path.stem() << ": Focal length is missing.\n";
                    }
                    else
                    {
                        const std::string cam_brand = exif_reader.getBrand();
                        const std::string cam_model = exif_reader.getModel();
                        if (!cam_brand.empty() && !cam_model.empty())
                        {
                            const std::string cam_full_model = std::format("{} {}", cam_brand, cam_model);

                            Datasheet datasheet;
                            if (getInfo(cam_full_model, vec_database, datasheet))
                            {
                                focal = std::max(width, height) * exif_reader.getFocal() / datasheet.sensorSize_;
                            }
                            else
                            {
                                std::cerr << image_file_path.stem() << "\" model \"" << cam_full_model
                                          << "\" doesn't exist in the database.\n"
                                          << "Please consider add your camera model and sensor width in the database.\n";
                            }
                        }
                    }
                }

                std::shared_ptr<IntrinsicBase> intrinsic;
                if ((focal > 0) && (ppx > 0) && (ppy > 0) && (width > 0) && (height > 0))
                {
                    intrinsic = std::make_shared<Pinhole_Intrinsic_Radial_K3>(
                        width, height, focal, ppx, ppy, 0.0, 0.0, 0.0); // setup no distortion as initial guess
                }

                {
                    const IndexT id = static_cast<IndexT>(views.size());
                    auto view = std::make_shared<openMVG::sfm::View>(iter->string(), id, id, id, width, height);

                    if (intrinsic)
                    {
                        intrinsics[view->id_intrinsic] = std::move(intrinsic);
                    }
                    else
                    {
                        // The view has a invalid intrinsic data. Still export the view, but keep the invalid intrinsic id.
                        view->id_intrinsic = UndefinedIndexT;
                    }

                    views[view->id_view] = std::move(view);
                }
            }

            // Group camera that share common properties, leads to more faster & stable BA.
            GroupSharedIntrinsics(sfm_data);

            return sfm_data;
        }

        FeatureRegions FeatureExtraction(const SfM_Data& sfm_data) const
        {
            // Reference from openMVG/src/software/SfM/main_ComputeFeatures.cpp

            features::SIFT_Anatomy_Image_describer image_describer;

            FeatureRegions feature_regions;
            feature_regions.regions_type = image_describer.Allocate();
            feature_regions.feature_regions.resize(sfm_data.views.size());

            const std::filesystem::path image_dir = sfm_data.s_root_path;

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
            for (int i = 0; i < static_cast<int>(sfm_data.views.size()); ++i)
            {
                Views::const_iterator iter = sfm_data.views.begin();
                std::advance(iter, i);

                const auto& view = *(iter->second);
                const auto view_file = image_dir / view.s_Img_path;

                Image<unsigned char> image_gray;
                if (!ReadImage(view_file.string().c_str(), &image_gray))
                {
                    continue;
                }

                feature_regions.feature_regions[i] = image_describer.Describe(image_gray);
            }

            return feature_regions;
        }

        PairWiseMatches PairMatching(const SfM_Data& sfm_data, FeatureRegions& regions) const
        {
            // Reference from openMVG/src/software/SfM/main_ComputeMatches.cpp

            // Load the corresponding view regions
            auto regions_provider = std::make_shared<Regions_Provider>();
            regions_provider->load(sfm_data, regions.feature_regions.data(), *regions.regions_type);

            const float dist_ratio = 0.8f;

            assert(regions.regions_type->IsScalar());
            Cascade_Hashing_Matcher_Regions matcher(dist_ratio);

            const Pair_Set pairs = exhaustivePairs(sfm_data.GetViews().size());
            std::cout << "Matching on # pairs: " << pairs.size() << '\n';

            PairWiseMatches map_putative_matches;
            matcher.Match(regions_provider, pairs, map_putative_matches);
            std::cout << "# putative pairs: " << map_putative_matches.size() << '\n';

            return map_putative_matches;
        }

        PairWiseMatches GeometricFilter(
            const SfM_Data& sfm_data, const PairWiseMatches& map_putative_matches, FeatureRegions& regions, bool sequential) const
        {
            // Reference from openMVG/src/software/SfM/main_GeometricFilter.cpp

            const bool guided_matching = false;
            const uint32_t max_iteration = 2048;

            auto regions_provider = std::make_shared<Regions_Provider>();
            regions_provider->load(sfm_data, regions.feature_regions.data(), *regions.regions_type);

            ImageCollectionGeometricFilter filter(&sfm_data, regions_provider);

            const double dist_ratio = 0.6;

            PairWiseMatches map_geometric_matches;
            if (sequential)
            {
                filter.Robust_model_estimation(
                    GeometricFilter_FMatrix_AC(4.0, max_iteration), map_putative_matches, guided_matching, dist_ratio);
                map_geometric_matches = filter.Get_geometric_matches();
            }
            else
            {
                filter.Robust_model_estimation(
                    GeometricFilter_EMatrix_AC(4.0, max_iteration), map_putative_matches, guided_matching, dist_ratio);
                map_geometric_matches = filter.Get_geometric_matches();

                for (auto iter = map_geometric_matches.begin(); iter != map_geometric_matches.end();)
                {
                    const size_t putative_photometric_count = map_putative_matches.find(iter->first)->second.size();
                    const size_t putative_geometric_count = iter->second.size();
                    const float ratio = putative_geometric_count / static_cast<float>(putative_photometric_count);
                    if ((putative_geometric_count < 50) || (ratio < 0.3f))
                    {
                        iter = map_geometric_matches.erase(iter);
                    }
                    else
                    {
                        ++iter;
                    }
                }
            }

            return map_geometric_matches;
        }

        SfM_Data PointCloudReconstruction(SfM_Data& sfm_data, const PairWiseMatches& map_geometric_matches, FeatureRegions& regions,
            bool sequential, const std::filesystem::path& tmp_dir) const
        {
            // Reference from openMVG/src/software/SfM/main_SfM.cpp

            Features_Provider feats_provider;
            feats_provider.load(sfm_data, regions.feature_regions.data());

            Matches_Provider matches_provider;
            matches_provider.load(sfm_data, map_geometric_matches);

            std::unique_ptr<ReconstructionEngine> sfm_engine;
            if (sequential)
            {
                auto engine = std::make_unique<SequentialSfMReconstructionEngine>(sfm_data, tmp_dir.string());

                engine->SetFeaturesProvider(&feats_provider);
                engine->SetMatchesProvider(&matches_provider);

                engine->SetUnknownCameraType(EINTRINSIC::PINHOLE_CAMERA_RADIAL3);
                engine->SetTriangulationMethod(ETriangulationMethod::DEFAULT);
                engine->SetResectionMethod(resection::SolverType::DEFAULT);

                sfm_engine = std::move(engine);
            }
            else
            {
                auto engine = std::make_unique<GlobalSfMReconstructionEngine_RelativeMotions>(sfm_data, tmp_dir.string());

                engine->SetFeaturesProvider(&feats_provider);
                engine->SetMatchesProvider(&matches_provider);

                engine->SetRotationAveragingMethod(ERotationAveragingMethod::ROTATION_AVERAGING_L2);
                engine->SetTranslationAveragingMethod(ETranslationAveragingMethod::TRANSLATION_AVERAGING_SOFTL1);

                sfm_engine = std::move(engine);
            }

            sfm_engine->Set_Intrinsics_Refinement_Type(Intrinsic_Parameter_Type::ADJUST_ALL);
            sfm_engine->Set_Extrinsics_Refinement_Type(Extrinsic_Parameter_Type::ADJUST_ALL);
            sfm_engine->Set_Use_Motion_Prior(false);

            if (!sfm_engine->Process())
            {
                throw std::runtime_error("Fail to process SfM.");
            }

            const SfM_Data processed_sfm_data = sfm_engine->Get_SfM_Data();
            Save(processed_sfm_data, (tmp_dir / "PointCloud_CameraPoses.ply").string(), ESfM_Data::ALL);

            return processed_sfm_data;
        }

        Result ExportResult(const SfM_Data& sfm_data)
        {
            // Reference from openMVG/src/software/SfM/export/main_openMVG2openMVS.cpp

            Result ret;

            ret.intrinsics.reserve(sfm_data.intrinsics.size());
            std::map<IndexT, uint32_t> intrinsic_id_mapping;
            for (const auto& mvg_intrinsic : sfm_data.intrinsics)
            {
                if (intrinsic_id_mapping.find(mvg_intrinsic.first) == intrinsic_id_mapping.end())
                {
                    intrinsic_id_mapping.emplace(mvg_intrinsic.first, static_cast<uint32_t>(ret.intrinsics.size()));
                }

                assert(dynamic_cast<const Pinhole_Intrinsic*>(mvg_intrinsic.second.get()) != nullptr);
                const Pinhole_Intrinsic& cam = static_cast<const Pinhole_Intrinsic&>(*mvg_intrinsic.second);

                auto& result_intrinsic = ret.intrinsics.emplace_back();
                result_intrinsic.width = cam.w();
                result_intrinsic.height = cam.h();
                result_intrinsic.k = cam.K();
            }

            GpuTexture2D distort_gpu_tex;
            GpuTexture2D undistort_gpu_tex;

            ret.views.reserve(sfm_data.views.size());
            const std::filesystem::path image_dir = sfm_data.s_root_path;
            std::map<IndexT, uint32_t> view_id_mapping;
            for (const auto& mvg_view : sfm_data.views)
            {
                const auto src_image = image_dir / mvg_view.second->s_Img_path;
                if (!std::filesystem::is_regular_file(src_image))
                {
                    throw std::runtime_error(std::format("Cannot read the corresponding image: {}", src_image.string()));
                }

                if (sfm_data.IsPoseAndIntrinsicDefined(mvg_view.second.get()))
                {
                    view_id_mapping.emplace(mvg_view.first, static_cast<uint32_t>(ret.views.size()));

                    auto& result_view = ret.views.emplace_back();
                    result_view.intrinsic_id = intrinsic_id_mapping.at(mvg_view.second->id_intrinsic);

                    const auto& mvg_pose = sfm_data.poses.at(mvg_view.second->id_pose);
                    result_view.rotation = mvg_pose.rotation();
                    result_view.center = mvg_pose.center();

                    result_view.image = LoadTexture(src_image);

                    const auto& camera = *sfm_data.intrinsics.at(mvg_view.second->id_intrinsic);
                    if (camera.have_disto())
                    {
                        Ensure4Channel(result_view.image);

                        if (!distort_gpu_tex || (distort_gpu_tex.Width(0) != result_view.image.Width()) ||
                            (distort_gpu_tex.Height(0) != result_view.image.Height()))
                        {
                            distort_gpu_tex = GpuTexture2D(gpu_system_, result_view.image.Width(), result_view.image.Height(), 1, ColorFmt,
                                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
                        }
                        if (!undistort_gpu_tex || (undistort_gpu_tex.Width(0) != result_view.image.Width()) ||
                            (undistort_gpu_tex.Height(0) != result_view.image.Height()))
                        {
                            undistort_gpu_tex = GpuTexture2D(gpu_system_, result_view.image.Width(), result_view.image.Height(), 1,
                                ColorFmt, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
                        }

                        auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

                        distort_gpu_tex.Upload(gpu_system_, cmd_list, 0, result_view.image.Data());

                        assert(dynamic_cast<const Pinhole_Intrinsic_Radial_K3*>(&camera) != nullptr);
                        Undistort(cmd_list, static_cast<const Pinhole_Intrinsic_Radial_K3&>(camera), distort_gpu_tex, undistort_gpu_tex);

                        if (result_view.image.NumChannels() != 4)
                        {
                            result_view.image =
                                Texture(undistort_gpu_tex.Width(0), undistort_gpu_tex.Height(0), FormatSize(undistort_gpu_tex.Format()));
                        }
                        undistort_gpu_tex.Readback(gpu_system_, cmd_list, 0, result_view.image.Data());

                        gpu_system_.Execute(std::move(cmd_list));
                        gpu_system_.WaitForGpu();
                    }
                }
                else
                {
                    std::cout << "Cannot read the corresponding pose or intrinsic of view " << mvg_view.first << '\n';
                }
            }

            ret.structure.reserve(sfm_data.GetLandmarks().size());
            for (const auto& mvg_vertex : sfm_data.GetLandmarks())
            {
                const auto& mvg_landmark = mvg_vertex.second;
                auto& result_landmark = ret.structure.emplace_back();
                result_landmark.point = mvg_landmark.X;
                for (const auto& mvg_observation : mvg_landmark.obs)
                {
                    const auto iter = view_id_mapping.find(mvg_observation.first);
                    if (iter != view_id_mapping.end())
                    {
                        auto& result_observation = result_landmark.obs.emplace_back();
                        result_observation.view_id = iter->second;
                        result_observation.point = mvg_observation.second.x;
                        result_observation.feat_id = mvg_observation.second.id_feat;
                    }
                }
            }

            distort_gpu_tex.Reset();
            undistort_gpu_tex.Reset();

            gpu_system_.WaitForGpu();

            return ret;
        }

        void Undistort(
            GpuCommandList& cmd_list, const Pinhole_Intrinsic_Radial_K3& camera, const GpuTexture2D& input_tex, GpuTexture2D& output_tex)
        {
            constexpr uint32_t BlockDim = 16;

            auto* d3d12_cmd_list = cmd_list.NativeCommandList<ID3D12GraphicsCommandList>();

            const auto& k = camera.K();
            undistort_cb_->k.x = static_cast<float>(k(0, 0));
            undistort_cb_->k.y = static_cast<float>(k(0, 2));
            undistort_cb_->k.z = static_cast<float>(k(1, 2));
            const auto& params = camera.getParams();
            undistort_cb_->params.x = static_cast<float>(params[3]);
            undistort_cb_->params.y = static_cast<float>(params[4]);
            undistort_cb_->params.z = static_cast<float>(params[5]);
            undistort_cb_->width_height.x = static_cast<float>(input_tex.Width(0));
            undistort_cb_->width_height.y = static_cast<float>(input_tex.Height(0));
            undistort_cb_->width_height.z = 1.0f / input_tex.Width(0);
            undistort_cb_->width_height.w = 1.0f / input_tex.Height(0);
            undistort_cb_.UploadToGpu();

            d3d12_cmd_list->SetPipelineState(undistort_pso_.Get());
            d3d12_cmd_list->SetComputeRootSignature(undistort_root_sig_.Get());

            auto srv_uav_desc_block = gpu_system_.AllocCbvSrvUavDescBlock(2);
            const uint32_t srv_descriptor_size = gpu_system_.CbvSrvUavDescSize();

            ID3D12DescriptorHeap* heaps[] = {srv_uav_desc_block.NativeDescriptorHeap()};
            d3d12_cmd_list->SetDescriptorHeaps(static_cast<uint32_t>(std::size(heaps)), heaps);

            input_tex.Transition(cmd_list, D3D12_RESOURCE_STATE_COMMON);
            output_tex.Transition(cmd_list, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

            GpuShaderResourceView srv(gpu_system_, input_tex, OffsetHandle(srv_uav_desc_block.CpuHandle(), 0, srv_descriptor_size));
            GpuUnorderedAccessView uav(gpu_system_, output_tex, OffsetHandle(srv_uav_desc_block.CpuHandle(), 1, srv_descriptor_size));

            d3d12_cmd_list->SetComputeRootConstantBufferView(0, undistort_cb_.GpuVirtualAddress());
            d3d12_cmd_list->SetComputeRootDescriptorTable(1, OffsetHandle(srv_uav_desc_block.GpuHandle(), 0, srv_descriptor_size));
            d3d12_cmd_list->SetComputeRootDescriptorTable(2, OffsetHandle(srv_uav_desc_block.GpuHandle(), 1, srv_descriptor_size));

            d3d12_cmd_list->Dispatch(DivUp(output_tex.Width(0), BlockDim), DivUp(output_tex.Height(0), BlockDim), 1);

            output_tex.Transition(cmd_list, D3D12_RESOURCE_STATE_COMMON);

            gpu_system_.DeallocCbvSrvUavDescBlock(std::move(srv_uav_desc_block));
        }

    private:
        const std::filesystem::path exe_dir_;

        GpuSystem& gpu_system_;

        struct UndistortConstantBuffer
        {
            XMFLOAT3 k;
            float padding_0;
            XMFLOAT3 params;
            float padding_1;
            XMFLOAT4 width_height;
        };
        ConstantBuffer<UndistortConstantBuffer> undistort_cb_;
        ComPtr<ID3D12RootSignature> undistort_root_sig_;
        ComPtr<ID3D12PipelineState> undistort_pso_;

        static constexpr DXGI_FORMAT ColorFmt = DXGI_FORMAT_R8G8B8A8_UNORM;
    };

    StructureFromMotion::StructureFromMotion(const std::filesystem::path& exe_dir, GpuSystem& gpu_system)
        : impl_(std::make_unique<Impl>(exe_dir, gpu_system))
    {
    }

    StructureFromMotion::~StructureFromMotion() noexcept = default;

    StructureFromMotion::StructureFromMotion(StructureFromMotion&& other) noexcept = default;
    StructureFromMotion& StructureFromMotion::operator=(StructureFromMotion&& other) noexcept = default;

    StructureFromMotion::Result StructureFromMotion::Process(
        const std::filesystem::path& image_dir, bool sequential, const std::filesystem::path& tmp_dir)
    {
        return impl_->Process(image_dir, sequential, tmp_dir);
    }
} // namespace AIHoloImager
