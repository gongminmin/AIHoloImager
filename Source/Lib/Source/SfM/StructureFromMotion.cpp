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
#ifdef AIHI_KEEP_INTERMEDIATES
    #include "AIHoloImager/Mesh.hpp"
#endif

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

                point_cloud_estimator_module_ = python_system.Import("PointCloudEstimator");
                point_cloud_estimator_class_ = python_system.GetAttr(*point_cloud_estimator_module_, "PointCloudEstimator");
                point_cloud_estimator_ = python_system.CallObject(*point_cloud_estimator_class_);
                point_cloud_estimator_focal_method_ = python_system.GetAttr(*point_cloud_estimator_, "Focal");
                point_cloud_estimator_point_cloud_method_ = python_system.GetAttr(*point_cloud_estimator_, "PointCloud");
            });

            auto& gpu_system = aihi_.GpuSystemInstance();

            const GpuStaticSampler bilinear_sampler(
                gpu_system, {GpuStaticSampler::Filter::Linear, GpuStaticSampler::Filter::Linear}, GpuStaticSampler::AddressMode::Clamp);

            const ShaderInfo shader = {DEFINE_SHADER(UndistortCs)};
            undistort_pipeline_ = GpuComputePipeline(gpu_system, shader, std::span(&bilinear_sampler, 1));
        }

        ~Impl()
        {
            PerfRegion destroy_perf(aihi_.PerfProfilerInstance(), "SfM destroy");

            py_init_future_.wait();

            PythonSystem::GilGuard guard;

            auto& python_system = aihi_.PythonSystemInstance();
            auto point_cloud_estimator_destroy_method = python_system.GetAttr(*point_cloud_estimator_, "Destroy");
            python_system.CallObject(*point_cloud_estimator_);

            point_cloud_estimator_destroy_method.reset();
            point_cloud_estimator_focal_method_.reset();
            point_cloud_estimator_point_cloud_method_.reset();
            point_cloud_estimator_.reset();
            point_cloud_estimator_class_.reset();
            point_cloud_estimator_module_.reset();
        }

        Result Process(const std::filesystem::path& input_path, bool sequential, bool no_delight)
        {
            PerfRegion process_perf(aihi_.PerfProfilerInstance(), "SfM process");

            const auto sfm_tmp_dir = aihi_.TmpDir() / "Sfm";
            std::filesystem::create_directories(sfm_tmp_dir);

            std::vector<Texture> images;
            SfM_Data sfm_data = this->IntrinsicAnalysis(input_path, images);
            if (images.size() > 1)
            {
                const FeatureRegions regions = this->FeatureExtraction(sfm_data, images);
                const PairWiseMatches map_putative_matches = this->PairMatching(sfm_data, regions);
                const PairWiseMatches map_geometric_matches = this->GeometricFilter(sfm_data, map_putative_matches, regions, sequential);

                sfm_data = this->PointCloudReconstruction(sfm_data, map_geometric_matches, regions, sequential, sfm_tmp_dir);
            }
            else
            {
                sfm_data.poses.emplace(0, geometry::Pose3());
            }

            return this->ExportResult(sfm_data, images, no_delight, sfm_tmp_dir);
        }

    private:
        SfM_Data IntrinsicAnalysis(const std::filesystem::path& input_path, std::vector<Texture>& images) const
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

            std::filesystem::path image_dir;
            std::vector<std::filesystem::path> input_image_paths;
            if (std::filesystem::is_directory(input_path))
            {
                image_dir = input_path;
                for (const auto& dir_entry : std::filesystem::directory_iterator{image_dir})
                {
                    if (dir_entry.is_regular_file())
                    {
                        input_image_paths.push_back(std::filesystem::relative(dir_entry.path(), image_dir).string());
                    }
                }
                std::sort(input_image_paths.begin(), input_image_paths.end());
            }
            else
            {
                image_dir = input_path.parent_path();
                input_image_paths.push_back(input_path.filename());
            }
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

                            auto py_focal = python_system.CallObject(*point_cloud_estimator_focal_method_, *args);
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
            std::cout << std::format("Matching on # pairs: {}\n", pairs.size());

            PairWiseMatches map_putative_matches;
            matcher.Match(regions_provider, pairs, map_putative_matches);
            std::cout << std::format("# putative pairs: {}\n", map_putative_matches.size());

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

        Result ExportResult(const SfM_Data& sfm_data, const std::vector<Texture>& images, bool no_delight,
            [[maybe_unused]] const std::filesystem::path& tmp_dir)
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
                            GpuTexture2D(gpu_system, image.Width(), image.Height(), 1, ColorFmt, GpuResourceFlag::None, "distort_gpu_tex");
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
                                GpuResourceFlag::UnorderedAccess, "undistort_gpu_tex");
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

                    GpuTexture2D roi_image = this->CropImage(cmd_list, *process_tex, roi, result_view.delighted_offset);
                    if (no_delight)
                    {
                        result_view.delighted_tex = std::move(roi_image);
                    }
                    else
                    {
                        result_view.delighted_tex = delighter_.Process(cmd_list, roi_image);
                    }
                    gpu_system.Execute(std::move(cmd_list));
                }
                else
                {
                    std::cout << std::format("Cannot read the corresponding pose or intrinsic of view {}\n", mvg_view.first);
                }
            }

            distort_gpu_tex = GpuTexture2D();
            undistort_gpu_tex = GpuTexture2D();

            const auto& sfm_landmarks = sfm_data.GetLandmarks();
            if (sfm_landmarks.empty())
            {
                {
                    PerfRegion wait_perf(aihi_.PerfProfilerInstance(), "Wait for init");
                    py_init_future_.wait();
                }
                {
                    PerfRegion point_cloud_perf(aihi_.PerfProfilerInstance(), "Gen point cloud");

                    const Texture& image = images[0];
                    const uint32_t width = image.Width();
                    const uint32_t height = image.Height();

                    PythonSystem::GilGuard guard;

                    auto& python_system = aihi_.PythonSystemInstance();
                    auto args = python_system.MakeTuple(5);
                    {
                        auto py_image = python_system.MakeObject(
                            std::span<const std::byte>(reinterpret_cast<const std::byte*>(image.Data()), image.DataSize()));
                        python_system.SetTupleItem(*args, 0, std::move(py_image));

                        python_system.SetTupleItem(*args, 1, python_system.MakeObject(width));
                        python_system.SetTupleItem(*args, 2, python_system.MakeObject(height));
                        python_system.SetTupleItem(*args, 3, python_system.MakeObject(FormatChannels(image.Format())));

                        const double fx = ret.intrinsics[0].k[0].x;
                        const float fov_x = glm::degrees(static_cast<float>(2 * std::atan2(width, 2 * fx)));
                        python_system.SetTupleItem(*args, 4, python_system.MakeObject(fov_x));
                    }

                    auto py_point_cloud_items = python_system.CallObject(*point_cloud_estimator_point_cloud_method_, *args);

                    const auto py_point_cloud = python_system.GetTupleItem(*py_point_cloud_items, 0);
                    const uint32_t point_cloud_width = python_system.Cast<uint32_t>(*python_system.GetTupleItem(*py_point_cloud_items, 1));
                    const uint32_t point_cloud_height = python_system.Cast<uint32_t>(*python_system.GetTupleItem(*py_point_cloud_items, 2));
                    const uint32_t point_cloud_size = point_cloud_width * point_cloud_height;

                    auto cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Render);
                    auto& tensor_converter = aihi_.TensorConverterInstance();
                    GpuBuffer point_cloud_buff;
                    tensor_converter.ConvertPy(
                        cmd_list, *py_point_cloud, point_cloud_buff, GpuHeap::Default, GpuResourceFlag::None, "point_cloud_buff");
                    auto point_cloud = std::make_unique<glm::vec3[]>(point_cloud_size);
                    const auto rb_future =
                        cmd_list.ReadBackAsync(point_cloud_buff, point_cloud.get(), point_cloud_size * sizeof(glm::vec3));
                    gpu_system.Execute(std::move(cmd_list));

                    rb_future.wait();

                    ret.structure.reserve(point_cloud_width * point_cloud_height);
                    for (uint32_t y = 0; y < point_cloud_height; ++y)
                    {
                        for (uint32_t x = 0; x < point_cloud_width; ++x)
                        {
                            const glm::vec3& p = point_cloud[y * point_cloud_width + x];
                            if (p.z > 0)
                            {
                                bool valid = true;
                                for (int32_t dy = -1; valid && (dy <= 1); ++dy)
                                {
                                    for (int32_t dx = -1; valid && (dx <= 1); ++dx)
                                    {
                                        if ((dx != 0) || (dy != 0))
                                        {
                                            const int32_t nx = static_cast<int32_t>(x) + dx;
                                            const int32_t ny = static_cast<int32_t>(y) + dy;
                                            if ((nx >= 0) && (nx < static_cast<int32_t>(point_cloud_width)) && (ny >= 0) &&
                                                (ny < static_cast<int32_t>(point_cloud_height)))
                                            {
                                                const glm::vec3& np = point_cloud[ny * point_cloud_width + nx];
                                                if ((np.z > 0) && (p.z - np.z > 0.05f))
                                                {
                                                    valid = false;
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }

                                if (valid)
                                {
                                    auto& result_landmark = ret.structure.emplace_back();
                                    result_landmark.point = {p.x, p.y, p.z};

                                    auto& result_observation = result_landmark.obs.emplace_back();
                                    result_observation.view_id = 0;
                                    result_observation.point = {
                                        (x + 0.5f) / point_cloud_width * width, (y + 0.5f) / point_cloud_height * height};
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                ret.structure.reserve(sfm_landmarks.size());
                for (const auto& mvg_vertex : sfm_landmarks)
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
                        }
                    }
                }
            }

#ifdef AIHI_KEEP_INTERMEDIATES
            {
                const VertexAttrib pos_clr_vertex_attribs[] = {
                    {VertexAttrib::Semantic::Position, 0, 3},
                    {VertexAttrib::Semantic::Color, 0, 3},
                };
                constexpr uint32_t PosAttribIndex = 0;
                constexpr uint32_t ColorAttribIndex = 1;
                const VertexDesc pos_clr_vertex_desc(pos_clr_vertex_attribs);

                {
                    Mesh pc_mesh = Mesh(pos_clr_vertex_desc, 0, 0);

                    for (const auto& landmark : ret.structure)
                    {
                        const uint32_t vertex_index = pc_mesh.NumVertices();
                        pc_mesh.ResizeVertices(vertex_index + 1);

                        pc_mesh.VertexData<glm::vec3>(vertex_index, PosAttribIndex) = glm::vec3(landmark.point);

                        const auto& ob = landmark.obs[0];
                        const uint32_t x = static_cast<uint32_t>(std::round(ob.point.x));
                        const uint32_t y = static_cast<uint32_t>(std::round(ob.point.y));
                        const auto& image = images[ob.view_id];
                        const std::byte* image_data = image.Data();
                        const uint32_t offset = (y * image.Width() + x) * FormatChannels(image.Format());

                        pc_mesh.VertexData<glm::vec3>(vertex_index, ColorAttribIndex) =
                            glm::vec3(image_data[offset + 0], image_data[offset + 1], image_data[offset + 2]) / 255.0f;
                    }

                    SaveMesh(pc_mesh, tmp_dir / "PointCloud.ply");
                }
            }
#endif

            return ret;
        }

        void Undistort(
            GpuCommandList& cmd_list, const Pinhole_Intrinsic_Radial_K3& camera, const GpuTexture2D& input_tex, GpuTexture2D& output_tex)
        {
            auto& gpu_system = aihi_.GpuSystemInstance();

            constexpr uint32_t BlockDim = 16;

            GpuConstantBufferOfType<UndistortConstantBuffer> undistort_cb(gpu_system, "undistort_cb");

            const auto& k = camera.K();
            undistort_cb->k.x = static_cast<float>(k(0, 0));
            undistort_cb->k.y = static_cast<float>(k(0, 2));
            undistort_cb->k.z = static_cast<float>(k(1, 2));
            const auto& params = camera.getParams();
            undistort_cb->params.x = static_cast<float>(params[3]);
            undistort_cb->params.y = static_cast<float>(params[4]);
            undistort_cb->params.z = static_cast<float>(params[5]);
            undistort_cb->width_height.x = static_cast<float>(input_tex.Width(0));
            undistort_cb->width_height.y = static_cast<float>(input_tex.Height(0));
            undistort_cb->width_height.z = 1.0f / input_tex.Width(0);
            undistort_cb->width_height.w = 1.0f / input_tex.Height(0);
            undistort_cb.UploadStaging();

            const GpuShaderResourceView input_srv(gpu_system, input_tex);
            GpuUnorderedAccessView output_uav(gpu_system, output_tex);

            std::tuple<std::string_view, const GpuConstantBuffer*> cbs[] = {
                {"param_cb", &undistort_cb},
            };
            std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                {"distorted_tex", &input_srv},
            };
            std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                {"undistorted_tex", &output_uav},
            };
            const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
            cmd_list.Compute(
                undistort_pipeline_, DivUp(output_tex.Width(0), BlockDim), DivUp(output_tex.Height(0), BlockDim), 1, shader_binding);
        }

        GpuTexture2D CropImage(GpuCommandList& cmd_list, const GpuTexture2D& image, const glm::uvec4& roi, glm::uvec2& offset)
        {
            constexpr uint32_t Gap = 32;
            glm::uvec4 expanded_roi;
            expanded_roi.x = std::max(static_cast<int32_t>(std::floor(roi.x) - Gap), 0);
            expanded_roi.y = std::max(static_cast<int32_t>(std::floor(roi.y) - Gap), 0);
            expanded_roi.z = std::min(static_cast<int32_t>(std::ceil(roi.z) + Gap), static_cast<int32_t>(image.Width(0)));
            expanded_roi.w = std::min(static_cast<int32_t>(std::ceil(roi.w) + Gap), static_cast<int32_t>(image.Height(0)));

            offset = glm::uvec2(expanded_roi.x, expanded_roi.y);

            auto& gpu_system = aihi_.GpuSystemInstance();

            const uint32_t roi_width = expanded_roi.z - expanded_roi.x;
            const uint32_t roi_height = expanded_roi.w - expanded_roi.y;
            GpuTexture2D roi_image(gpu_system, roi_width, roi_height, 1, image.Format(),
                GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable, "roi_image");
            cmd_list.Copy(roi_image, 0, 0, 0, 0, image, 0, GpuBox{expanded_roi.x, expanded_roi.y, 0, expanded_roi.z, expanded_roi.w, 1});

            return roi_image;
        }

    private:
        AIHoloImagerInternal& aihi_;

        PyObjectPtr point_cloud_estimator_module_;
        PyObjectPtr point_cloud_estimator_class_;
        PyObjectPtr point_cloud_estimator_;
        PyObjectPtr point_cloud_estimator_focal_method_;
        PyObjectPtr point_cloud_estimator_point_cloud_method_;
        std::future<void> py_init_future_;

        struct UndistortConstantBuffer
        {
            glm::vec3 k;
            float padding_0;
            glm::vec3 params;
            float padding_1;
            glm::vec4 width_height;
        };
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

    StructureFromMotion::Result StructureFromMotion::Process(const std::filesystem::path& image_dir, bool sequential, bool no_delight)
    {
        return impl_->Process(image_dir, sequential, no_delight);
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
        const float fov = static_cast<float>(2 * std::atan2(intrinsic.height, 2 * fy));
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
