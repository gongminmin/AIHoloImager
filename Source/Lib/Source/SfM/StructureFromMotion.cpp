// Copyright (c) 2024-2025 Minmin Gong
//

#include "StructureFromMotion.hpp"

#include <algorithm>
#include <format>
#include <future>
#include <map>
#include <stdexcept>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

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
    #pragma warning(disable : 4305) // Ignore truncation from double to float
    #pragma warning(disable : 4702) // Ignore unreachable code
    #pragma warning(disable : 5054) // Ignore operator between enums of different types
    #pragma warning(disable : 5055) // Ignore operator between enums and floating-point types
#endif
#include <openMVG/cameras/Camera_Pinhole_Radial.hpp>
#include <openMVG/exif/exif_IO_EasyExif.hpp>
#include <openMVG/exif/sensor_width_database/datasheet.hpp>
#include <openMVG/features/regions.hpp>
#include <openMVG/features/sift/SIFT_Anatomy_Image_Describer.hpp>
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
#ifdef AIHI_KEEP_INTERMEDIATES
    #include <openMVG/sfm/sfm_data_io.hpp>
#endif
#ifdef _MSC_VER
    #pragma warning(pop)
#endif

#include "Delighter/Delighter.hpp"
#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuConstantBuffer.hpp"
#include "Gpu/GpuSampler.hpp"
#include "Gpu/GpuShader.hpp"
#include "Gpu/GpuTexture.hpp"
#include "MaskGen/MaskGenerator.hpp"
#include "Util/PerfProfiler.hpp"

#include "CompiledShader/SfM/UndistortCs.h"

using namespace openMVG;
using namespace openMVG::cameras;
using namespace openMVG::sfm;
using namespace openMVG::image;
using namespace openMVG::matching;
using namespace openMVG::matching_image_collection;

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
        explicit Impl(AIHoloImagerInternal& aihi) : aihi_(aihi), mask_gen_(aihi), delighter_(aihi)
        {
            PerfRegion init_perf(aihi_.PerfProfilerInstance(), "SfM init");

            py_init_future_ = std::async(std::launch::async, [this] {
                PerfRegion init_async_perf(aihi_.PerfProfilerInstance(), "Focal estimator init (async)");

                PythonSystem::GilGuard guard;

                auto& python_system = aihi_.PythonSystemInstance();

                focal_estimator_module_ = python_system.Import("FocalEstimator");
                focal_estimator_class_ = python_system.GetAttr(*focal_estimator_module_, "FocalEstimator");
                focal_estimator_ = python_system.CallObject(*focal_estimator_class_);
                focal_estimator_process_method_ = python_system.GetAttr(*focal_estimator_, "Process");
            });

            auto& gpu_system = aihi_.GpuSystemInstance();

            undistort_cb_ = GpuConstantBufferOfType<UndistortConstantBuffer>(gpu_system, L"undistort_cb_");

            const GpuStaticSampler bilinear_sampler(
                {GpuStaticSampler::Filter::Linear, GpuStaticSampler::Filter::Linear}, GpuStaticSampler::AddressMode::Clamp);

            const ShaderInfo shader = {UndistortCs_shader, 1, 1, 1};
            undistort_pipeline_ = GpuComputePipeline(gpu_system, shader, std::span(&bilinear_sampler, 1));
        }

        ~Impl()
        {
            PerfRegion destroy_perf(aihi_.PerfProfilerInstance(), "SfM destroy");

            py_init_future_.wait();

            PythonSystem::GilGuard guard;

            auto& python_system = aihi_.PythonSystemInstance();
            auto focal_estimator_destroy_method = python_system.GetAttr(*focal_estimator_, "Destroy");
            python_system.CallObject(*focal_estimator_destroy_method);

            focal_estimator_destroy_method.reset();
            focal_estimator_process_method_.reset();
            focal_estimator_.reset();
            focal_estimator_class_.reset();
            focal_estimator_module_.reset();
        }

        Result Process(const std::filesystem::path& input_path, bool sequential)
        {
            PerfRegion process_perf(aihi_.PerfProfilerInstance(), "SfM process");

            std::vector<Texture> images;
            SfM_Data sfm_data = this->IntrinsicAnalysis(input_path, images);
            FeatureRegions regions = this->FeatureExtraction(sfm_data, images);
            const PairWiseMatches map_putative_matches = this->PairMatching(sfm_data, regions);
            const PairWiseMatches map_geometric_matches = this->GeometricFilter(sfm_data, map_putative_matches, regions, sequential);

            const auto sfm_tmp_dir = aihi_.TmpDir() / "Sfm";
            std::filesystem::create_directories(sfm_tmp_dir);

            const SfM_Data processed_sfm_data =
                this->PointCloudReconstruction(sfm_data, map_geometric_matches, regions, sequential, sfm_tmp_dir);
            return this->ExportResult(processed_sfm_data, images);
        }

    private:
        SfM_Data IntrinsicAnalysis(const std::filesystem::path& image_dir, std::vector<Texture>& images) const
        {
            // Reference from openMVG/src/software/SfM/main_SfMInit_ImageListing.cpp

            PerfRegion process_perf(aihi_.PerfProfilerInstance(), "Intrinsic analysis");

            const auto camera_sensor_db_path = aihi_.ExeDir() / "CameraDatabase.dat";
            std::vector<Datasheet> vec_database;
            {
                std::ifstream db_file(camera_sensor_db_path.string(), std::ios::binary);
                if (db_file.is_open())
                {
                    uint32_t num_datasheets = 0;
                    db_file.read(reinterpret_cast<char*>(&num_datasheets), sizeof(num_datasheets));
                    vec_database.resize(num_datasheets);
                    for (uint32_t i = 0; i < num_datasheets; ++i)
                    {
                        uint16_t model_len = 0;
                        db_file.read(reinterpret_cast<char*>(&model_len), sizeof(model_len));
                        std::string model(model_len, '\0');
                        db_file.read(model.data(), model_len);

                        float sensor_size = 0;
                        db_file.read(reinterpret_cast<char*>(&sensor_size), sizeof(sensor_size));

                        vec_database[i] = {std::move(model), sensor_size};
                    }
                }
                else
                {
                    throw std::runtime_error(
                        std::format("Invalid input database: {}, please specify a valid file.", camera_sensor_db_path.string()));
                }
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
            images.resize(input_image_paths.size());

            struct CameraInfo
            {
                std::string full_model;
                uint32_t width = 0;
                uint32_t height = 0;
                double focal = -1;

                std::vector<size_t> image_ids;
            };

            std::vector<CameraInfo> camera_infos;
            {
                exif::Exif_IO_EasyExif exif_reader;
                for (size_t i = 0; i < input_image_paths.size(); ++i)
                {
                    const std::filesystem::path image_file_path = image_dir / input_image_paths[i];

                    auto& image = images[i];
                    image = LoadTexture(image_file_path);
                    image.Convert(ElementFormat::RGBA8_UNorm);

                    CameraInfo camera_info = {"unknown Unknown", image.Width(), image.Height()};
                    if (exif_reader.open(image_file_path.string()))
                    {
                        if (exif_reader.doesHaveExifInfo())
                        {
                            const double focal = exif_reader.getFocal();
                            if (focal != 0.0)
                            {
                                const std::string cam_brand = exif_reader.getBrand();
                                const std::string cam_model = exif_reader.getModel();
                                if (!cam_brand.empty() && !cam_model.empty())
                                {
                                    camera_info.full_model = std::format("{} {}", cam_brand, cam_model);
                                    camera_info.focal = focal;
                                }
                            }
                            else
                            {
                                std::cerr << image_file_path.filename() << ": Focal length is missing.\n";
                            }
                        }
                        else
                        {
                            std::cerr << image_file_path.filename() << ": Exif info is missing.\n";
                        }
                    }
                    else
                    {
                        std::cerr << image_file_path.filename() << ": Unable to read Exif info.\n";
                    }

                    // Group camera that share common properties, leads to more faster & stable BA.
                    bool found = false;
                    for (size_t j = 0; j < camera_infos.size(); ++j)
                    {
                        if ((camera_infos[j].full_model == camera_info.full_model) && (camera_infos[j].width == camera_info.width) &&
                            (camera_infos[j].height == camera_info.height) && (std::abs(camera_infos[j].focal - camera_info.focal) < 1e-6f))
                        {
                            camera_infos[j].image_ids.push_back(i);
                            found = true;
                            break;
                        }
                    }

                    if (!found)
                    {
                        camera_info.image_ids.push_back(i);
                        camera_infos.emplace_back(std::move(camera_info));
                    }
                }
            }

            SfM_Data sfm_data;
            sfm_data.s_root_path = image_dir.string();

            Views& views = sfm_data.views;
            for (size_t i = 0; i < input_image_paths.size(); ++i)
            {
                const IndexT id = static_cast<IndexT>(views.size());
                views[id] = std::make_shared<openMVG::sfm::View>(
                    input_image_paths[i].string(), id, UndefinedIndexT, id, images[i].Width(), images[i].Height());
            }

            Intrinsics& intrinsics = sfm_data.intrinsics;
            for (uint32_t i = 0; i < camera_infos.size(); ++i)
            {
                const auto& camera_info = camera_infos[i];
                const uint32_t width = camera_info.width;
                const uint32_t height = camera_info.height;

                double focal = -1;
                if (camera_info.focal > 0)
                {
                    const Datasheet ref_datasheet(camera_info.full_model, -1.0);
                    auto db_iter = std::find(vec_database.begin(), vec_database.end(), ref_datasheet);
                    if (db_iter != vec_database.end())
                    {
                        focal = std::max(width, height) * camera_info.focal / db_iter->sensorSize_;
                    }
                    else
                    {
                        std::cout << std::format("Model \"{}\" doesn't exist in the database. ", camera_info.full_model);
                    }
                }

                if (focal <= 0)
                {
                    std::cout << "Estimating the camera focal length...\n";

                    {
                        PerfRegion wait_perf(aihi_.PerfProfilerInstance(), "Wait for init");
                        py_init_future_.wait();
                    }

                    double sum_focal = 0;
                    for (const uint32_t image_id : camera_info.image_ids)
                    {
                        const auto& image = images[image_id];
                        {
                            PythonSystem::GilGuard guard;

                            auto& python_system = aihi_.PythonSystemInstance();
                            auto args = python_system.MakeTuple(4);
                            {
                                auto py_image = python_system.MakeObject(
                                    std::span<const std::byte>(reinterpret_cast<const std::byte*>(image.Data()), image.DataSize()));
                                python_system.SetTupleItem(*args, 0, std::move(py_image));

                                python_system.SetTupleItem(*args, 1, python_system.MakeObject(image.Width()));
                                python_system.SetTupleItem(*args, 2, python_system.MakeObject(image.Height()));
                                python_system.SetTupleItem(*args, 3, python_system.MakeObject(FormatChannels(image.Format())));
                            }

                            auto py_focal = python_system.CallObject(*focal_estimator_process_method_, *args);
                            sum_focal += python_system.Cast<double>(*py_focal);
                        }
                    }

                    focal = sum_focal / camera_info.image_ids.size();

                    if (focal > 0)
                    {
                        std::cout << std::format("Estimated as {:.3f} pixels.\n", focal);
                    }
                    else
                    {
                        std::cerr << std::format("Fail to estimate the focal length of Model \"{}\"\n", camera_info.full_model);
                    }
                }

                if (focal > 0)
                {
                    const double ppx = width / 2.0;
                    const double ppy = height / 2.0;
                    intrinsics[i] = std::make_shared<Pinhole_Intrinsic_Radial_K3>(
                        width, height, focal, ppx, ppy, 0.0, 0.0, 0.0); // setup no distortion as initial guess

                    for (const uint32_t image_id : camera_info.image_ids)
                    {
                        views[image_id]->id_intrinsic = i;
                    }
                }
            }

            return sfm_data;
        }

        FeatureRegions FeatureExtraction(const SfM_Data& sfm_data, const std::vector<Texture>& images) const
        {
            // Reference from openMVG/src/software/SfM/main_ComputeFeatures.cpp

            PerfRegion process_perf(aihi_.PerfProfilerInstance(), "Feature extraction");

            features::SIFT_Anatomy_Image_describer image_describer;

            FeatureRegions feature_regions;
            feature_regions.regions_type = image_describer.Allocate();
            feature_regions.feature_regions.resize(sfm_data.views.size());

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
            for (int i = 0; i < static_cast<int>(sfm_data.views.size()); ++i)
            {
                const auto& image = images[i];
                const uint8_t* image_data = reinterpret_cast<const uint8_t*>(image.Data());
                const uint32_t width = image.Width();
                const uint32_t height = image.Height();
                assert(image.Format() == ElementFormat::RGBA8_UNorm);

                Image<unsigned char> image_gray(width, height);
                for (uint32_t y = 0; y < height; ++y)
                {
                    for (uint32_t x = 0; x < width; ++x)
                    {
                        const uint32_t offset = (y * width + x) * 4;
                        const uint8_t r = image_data[offset + 0];
                        const uint8_t g = image_data[offset + 1];
                        const uint8_t b = image_data[offset + 2];
                        image_gray(y, x) = static_cast<uint8_t>(std::clamp(static_cast<int>(0.299f * r + 0.587f * g + 0.114f * b), 0, 255));
                    }
                }

                feature_regions.feature_regions[i] = image_describer.Describe(image_gray);
            }

            return feature_regions;
        }

        PairWiseMatches PairMatching(const SfM_Data& sfm_data, const FeatureRegions& regions) const
        {
            // Reference from openMVG/src/software/SfM/main_ComputeMatches.cpp

            PerfRegion process_perf(aihi_.PerfProfilerInstance(), "Pair matching");

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
            const SfM_Data& sfm_data, const PairWiseMatches& map_putative_matches, const FeatureRegions& regions, bool sequential) const
        {
            // Reference from openMVG/src/software/SfM/main_GeometricFilter.cpp

            PerfRegion process_perf(aihi_.PerfProfilerInstance(), "Geometric filter");

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

        SfM_Data PointCloudReconstruction(const SfM_Data& sfm_data, const PairWiseMatches& map_geometric_matches,
            const FeatureRegions& regions, bool sequential, const std::filesystem::path& tmp_dir) const
        {
            // Reference from openMVG/src/software/SfM/main_SfM.cpp

            PerfRegion process_perf(aihi_.PerfProfilerInstance(), "Point cloud reconstruction");

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
#ifdef AIHI_KEEP_INTERMEDIATES
            Save(processed_sfm_data, (tmp_dir / "PointCloud_CameraPoses.ply").string(), ESfM_Data::ALL);
#endif

            return processed_sfm_data;
        }

        Result ExportResult(const SfM_Data& sfm_data, const std::vector<Texture>& images)
        {
            // Reference from openMVG/src/software/SfM/export/main_openMVG2openMVS.cpp

            PerfRegion process_perf(aihi_.PerfProfilerInstance(), "Export result");

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
                const auto& k = cam.K();
                std::memcpy(&result_intrinsic.k, k.data(), sizeof(result_intrinsic.k));
                result_intrinsic.k = glm::transpose(result_intrinsic.k);
            }

            auto& gpu_system = aihi_.GpuSystemInstance();

            GpuTexture2D distort_gpu_tex;
            GpuTexture2D undistort_gpu_tex;

            ret.views.reserve(sfm_data.views.size());
            std::map<IndexT, uint32_t> view_id_mapping;
            for (const auto& mvg_view : sfm_data.views)
            {
                if (sfm_data.IsPoseAndIntrinsicDefined(mvg_view.second.get()))
                {
                    view_id_mapping.emplace(mvg_view.first, static_cast<uint32_t>(ret.views.size()));

                    auto& result_view = ret.views.emplace_back();
                    result_view.intrinsic_id = intrinsic_id_mapping.at(mvg_view.second->id_intrinsic);

                    const auto& mvg_pose = sfm_data.poses.at(mvg_view.second->id_pose);
                    const auto& rotation = mvg_pose.rotation();
                    std::memcpy(&result_view.rotation, rotation.data(), sizeof(result_view.rotation));
                    result_view.rotation = glm::transpose(result_view.rotation);
                    const auto& center = mvg_pose.center();
                    result_view.center = {center.x(), center.y(), center.z()};

                    const Texture& image = images[mvg_view.first];

                    if (!distort_gpu_tex || (distort_gpu_tex.Width(0) != image.Width()) || (distort_gpu_tex.Height(0) != image.Height()))
                    {
                        distort_gpu_tex =
                            GpuTexture2D(gpu_system, image.Width(), image.Height(), 1, ColorFmt, GpuResourceFlag::None, L"distort_gpu_tex");
                    }

                    auto cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Render);

                    cmd_list.Upload(
                        distort_gpu_tex, 0, [&image](void* dst_data, uint32_t row_pitch, [[maybe_unused]] uint32_t slice_pitch) {
                            const uint32_t width = image.Width();
                            const uint32_t height = image.Height();
                            const uint32_t src_channels = FormatChannels(image.Format());
                            const uint32_t dst_channels = FormatChannels(ColorFmt);
                            const std::byte* src = image.Data();
                            std::byte* dst = reinterpret_cast<std::byte*>(dst_data);
                            for (uint32_t y = 0; y < height; ++y)
                            {
                                for (uint32_t x = 0; x < width; ++x)
                                {
                                    std::memcpy(&dst[y * row_pitch + x * dst_channels], &src[(y * width + x) * src_channels], src_channels);
                                }
                            }
                        });

                    GpuTexture2D* process_tex;
                    const auto& camera = *sfm_data.intrinsics.at(mvg_view.second->id_intrinsic);
                    if (camera.have_disto())
                    {
                        if (!undistort_gpu_tex || (undistort_gpu_tex.Width(0) != image.Width()) ||
                            (undistort_gpu_tex.Height(0) != image.Height()))
                        {
                            undistort_gpu_tex = GpuTexture2D(gpu_system, image.Width(), image.Height(), 1, ColorFmt,
                                GpuResourceFlag::UnorderedAccess, L"undistort_gpu_tex");
                        }

                        assert(dynamic_cast<const Pinhole_Intrinsic_Radial_K3*>(&camera) != nullptr);
                        Undistort(cmd_list, static_cast<const Pinhole_Intrinsic_Radial_K3&>(camera), distort_gpu_tex, undistort_gpu_tex);

                        process_tex = &undistort_gpu_tex;
                    }
                    else
                    {
                        process_tex = &distort_gpu_tex;
                    }

                    glm::uvec4 roi;
                    mask_gen_.Generate(cmd_list, *process_tex, roi);

                    result_view.delighted_tex = delighter_.Process(cmd_list, *process_tex, roi, result_view.delighted_offset);
                    gpu_system.Execute(std::move(cmd_list));
                }
                else
                {
                    std::cout << "Cannot read the corresponding pose or intrinsic of view " << mvg_view.first << '\n';
                }
            }

            distort_gpu_tex = GpuTexture2D();
            undistort_gpu_tex = GpuTexture2D();

            ret.structure.reserve(sfm_data.GetLandmarks().size());
            for (const auto& mvg_vertex : sfm_data.GetLandmarks())
            {
                const auto& mvg_landmark = mvg_vertex.second;
                auto& result_landmark = ret.structure.emplace_back();
                result_landmark.point = {mvg_landmark.X.x(), mvg_landmark.X.y(), mvg_landmark.X.z()};
                for (const auto& mvg_observation : mvg_landmark.obs)
                {
                    const auto iter = view_id_mapping.find(mvg_observation.first);
                    if (iter != view_id_mapping.end())
                    {
                        auto& result_observation = result_landmark.obs.emplace_back();
                        result_observation.view_id = iter->second;
                        result_observation.point = {mvg_observation.second.x.x(), mvg_observation.second.x.y()};
                        result_observation.feat_id = mvg_observation.second.id_feat;
                    }
                }
            }

            return ret;
        }

        void Undistort(
            GpuCommandList& cmd_list, const Pinhole_Intrinsic_Radial_K3& camera, const GpuTexture2D& input_tex, GpuTexture2D& output_tex)
        {
            auto& gpu_system = aihi_.GpuSystemInstance();

            constexpr uint32_t BlockDim = 16;

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
            undistort_cb_.UploadStaging();

            GpuShaderResourceView input_srv(gpu_system, input_tex);
            GpuUnorderedAccessView output_uav(gpu_system, output_tex);

            const GpuConstantBuffer* cbs[] = {&undistort_cb_};
            const GpuShaderResourceView* srvs[] = {&input_srv};
            GpuUnorderedAccessView* uavs[] = {&output_uav};
            const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
            cmd_list.Compute(
                undistort_pipeline_, DivUp(output_tex.Width(0), BlockDim), DivUp(output_tex.Height(0), BlockDim), 1, shader_binding);
        }

    private:
        AIHoloImagerInternal& aihi_;

        PyObjectPtr focal_estimator_module_;
        PyObjectPtr focal_estimator_class_;
        PyObjectPtr focal_estimator_;
        PyObjectPtr focal_estimator_process_method_;
        std::future<void> py_init_future_;

        struct UndistortConstantBuffer
        {
            glm::vec3 k;
            float padding_0;
            glm::vec3 params;
            float padding_1;
            glm::vec4 width_height;
        };
        GpuConstantBufferOfType<UndistortConstantBuffer> undistort_cb_;
        GpuComputePipeline undistort_pipeline_;

        MaskGenerator mask_gen_;
        Delighter delighter_;

        static constexpr GpuFormat ColorFmt = GpuFormat::RGBA8_UNorm;
    };

    StructureFromMotion::StructureFromMotion(AIHoloImagerInternal& aihi) : impl_(std::make_unique<Impl>(aihi))
    {
    }

    StructureFromMotion::~StructureFromMotion() noexcept = default;

    StructureFromMotion::StructureFromMotion(StructureFromMotion&& other) noexcept = default;
    StructureFromMotion& StructureFromMotion::operator=(StructureFromMotion&& other) noexcept = default;

    StructureFromMotion::Result StructureFromMotion::Process(const std::filesystem::path& image_dir, bool sequential)
    {
        return impl_->Process(image_dir, sequential);
    }

    glm::mat4x4 CalcViewMatrix(const StructureFromMotion::View& view)
    {
        const glm::vec3 camera_pos = view.center;
        const glm::vec3 camera_up_vec = -view.rotation[1];
        const glm::vec3 camera_forward_vec = view.rotation[2];
        return glm::lookAtRH(camera_pos, camera_pos + camera_forward_vec, camera_up_vec);
    }

    glm::mat4x4 CalcProjMatrix(const StructureFromMotion::PinholeIntrinsic& intrinsic, float near_plane, float far_plane)
    {
        const double fy = intrinsic.k[1].y;
        const float fov = static_cast<float>(2 * std::atan(intrinsic.height / (2 * fy)));
        return glm::perspectiveRH_ZO(fov, static_cast<float>(intrinsic.width) / intrinsic.height, near_plane, far_plane);
    }

    glm::vec2 CalcNearFarPlane(const glm::mat4x4& view_mtx, const Obb& obb)
    {
        glm::vec3 corners[8];
        Obb::GetCorners(obb, corners);

        const glm::vec4 z_col(view_mtx[0].z, view_mtx[1].z, view_mtx[2].z, view_mtx[3].z);

        float min_z_es = std::numeric_limits<float>::max();
        float max_z_es = std::numeric_limits<float>::lowest();
        for (const auto& corner : corners)
        {
            const glm::vec4 pos(corner.x, corner.y, corner.z, 1);
            const float z = glm::dot(pos, z_col);
            min_z_es = std::min(min_z_es, z);
            max_z_es = std::max(max_z_es, z);
        }

        const float center_es_z = (max_z_es + min_z_es) / 2;
        const float extent_es_z = (max_z_es - min_z_es) / 2 * 1.05f;

        const float near_plane = center_es_z + extent_es_z;
        const float far_plane = center_es_z - extent_es_z;
        return glm::vec2(-near_plane, -far_plane);
    }

    glm::vec2 CalcViewportOffset(const StructureFromMotion::PinholeIntrinsic& intrinsic)
    {
        return {
            intrinsic.k[0].z - intrinsic.width / 2,
            intrinsic.k[1].z - intrinsic.height / 2,
        };
    }
} // namespace AIHoloImager
