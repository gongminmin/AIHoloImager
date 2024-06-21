// Copyright (c) 2024 Minmin Gong
//

#include "StructureFromMotion.hpp"

#include <algorithm>
#include <format>
#include <stdexcept>

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#endif

#ifndef OPENMVG_USE_OPENMP
    #define OPENMVG_USE_OPENMP
#endif

#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4100) // Ignore unreferenced formal parameter
    #pragma warning(disable : 4127) // Ignore conditional expression is constant
    #pragma warning(disable : 4244) // Ignore implicit conversion
    #pragma warning(disable : 4267) // Ignore implicit conversion
    #pragma warning(disable : 4702) // Ignore unreachable code
    #pragma warning(disable : 5054) // Ignore operator between enums of different types
#endif
#include <openMVG/cameras/Camera_Pinhole_Radial.hpp>
#include <openMVG/exif/exif_IO_EasyExif.hpp>
#include <openMVG/exif/sensor_width_database/ParseDatabase.hpp>
#include <openMVG/features/regions.hpp>
#include <openMVG/features/sift/SIFT_Anatomy_Image_Describer.hpp>
#include <openMVG/image/image_io.hpp>
#include <openMVG/matching_image_collection/Cascade_Hashing_Matcher_Regions.hpp>
#include <openMVG/matching_image_collection/Pair_Builder.hpp>
#include <openMVG/sfm/pipelines/sfm_regions_provider.hpp>
#include <openMVG/sfm/sfm_data.hpp>
#include <openMVG/sfm/sfm_data_utils.hpp>
#ifdef _MSC_VER
    #pragma warning(pop)
#endif

using namespace openMVG;
using namespace openMVG::cameras;
using namespace openMVG::sfm;
using namespace openMVG::image;
using namespace openMVG::matching;

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
        void Process(const std::filesystem::path& input_path)
        {
            SfM_Data sfm_data = this->IntrinsicAnalysis(input_path);
            FeatureRegions regions = this->FeatureExtraction(sfm_data);
            const PairWiseMatches map_putative_matches = this->PairMatching(sfm_data, regions);
        }

    private:
        SfM_Data IntrinsicAnalysis(const std::filesystem::path& image_dir) const
        {
            // Reference from openMVG/src/software/SfM/main_SfMInit_ImageListing.cpp

            char exe_path[MAX_PATH];
            GetModuleFileNameA(nullptr, exe_path, sizeof(exe_path));
            const auto camera_sensor_db_path = std::filesystem::path(exe_path).parent_path() / "sensor_width_camera_database.txt";

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

            openMVG::exif::Exif_IO_EasyExif exif_reader;
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
                    const openMVG::IndexT id = static_cast<openMVG::IndexT>(views.size());
                    auto view = std::make_shared<View>(iter->string(), id, id, id, width, height);

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
            // Reference from openMVG/src/software/SfM/openMVG_main_ComputeFeatures.cpp

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

                const View& view = *(iter->second);
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
            // Reference from openMVG/src/software/SfM/openMVG_main_ComputeMatches.cpp

            // Load the corresponding view regions
            auto regions_provider = std::make_shared<Regions_Provider>();
            regions_provider->load(sfm_data, regions.feature_regions.data(), *regions.regions_type);

            const float dist_ratio = 0.8f;

            assert(regions.regions_type->IsScalar());
            matching_image_collection::Cascade_Hashing_Matcher_Regions matcher(dist_ratio);

            const Pair_Set pairs = exhaustivePairs(sfm_data.GetViews().size());
            std::cout << "Matching on # pairs: " << pairs.size() << '\n';

            PairWiseMatches map_putative_matches;
            matcher.Match(regions_provider, pairs, map_putative_matches);
            std::cout << "# putative pairs: " << map_putative_matches.size() << '\n';

            return map_putative_matches;
        }
    };

    StructureFromMotion::StructureFromMotion() : impl_(std::make_unique<Impl>())
    {
    }

    StructureFromMotion::~StructureFromMotion() noexcept = default;

    StructureFromMotion::StructureFromMotion(StructureFromMotion&& other) noexcept = default;
    StructureFromMotion& StructureFromMotion::operator=(StructureFromMotion&& other) noexcept = default;

    void StructureFromMotion::Process(const std::filesystem::path& image_dir)
    {
        return impl_->Process(image_dir);
    }
} // namespace AIHoloImager
