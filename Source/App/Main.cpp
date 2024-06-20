// Copyright (c) 2024 Minmin Gong
//

#include <chrono>
#include <filesystem>
#include <format>
#include <iostream>

#ifndef _DEBUG
    #define CXXOPTS_NO_RTTI
#endif
#include <cxxopts.hpp>

#include "AIHoloImager/AIHoloImager.hpp"
#include "AIHoloImager/Texture.hpp"

int main(int argc, char* argv[])
{
    cxxopts::Options options("AIHoloImager", "AIHoloImager: AI generated mesh from photos.");
    // clang-format off
    options.add_options()
        ("H,help", "Produce help message.")
        ("I,input-path", "The directory that contains the input image sequence.", cxxopts::value<std::string>())
        ("O,output-path", "The path of the output mesh (\"<input-dir>/Output/Mesh.glb\" by default).", cxxopts::value<std::string>())
        ("v,version", "Version.");
    // clang-format on

    const auto vm = options.parse(argc, argv);

    if ((argc <= 1) || (vm.count("help") > 0))
    {
        std::cout << std::format("{}\n", options.help());
        return 0;
    }
    if (vm.count("version") > 0)
    {
        std::cout << "AIHoloImager, Version 0.1.0\n";
        return 0;
    }

    std::filesystem::path input_path;
    if (vm.count("input-path") > 0)
    {
        input_path = vm["input-path"].as<std::string>();
    }
    else
    {
        std::cerr << std::format("ERROR: MUST have a input path\n");
        return 1;
    }

    if (!std::filesystem::exists(input_path))
    {
        std::cerr << std::format("ERROR: COULDN'T find {}\n", input_path.string());
        return 1;
    }

    std::filesystem::path output_path;
    if (vm.count("output-path") > 0)
    {
        output_path = vm["output-path"].as<std::string>();
    }
    else
    {
        output_path = input_path / "Output/Mesh.glb";
    }

    std::filesystem::create_directories(output_path.parent_path());

    std::vector<std::filesystem::path> input_image_paths;
    for (const auto& dir_entry : std::filesystem::directory_iterator{input_path})
    {
        if (dir_entry.is_regular_file())
        {
            input_image_paths.push_back(dir_entry.path());
        }
    }
    std::sort(input_image_paths.begin(), input_image_paths.end());

    std::vector<AIHoloImager::Texture> input_images;
    for (const auto& image_path : input_image_paths)
    {
        input_images.push_back(AIHoloImager::LoadTexture(image_path));
    }

    auto start = std::chrono::high_resolution_clock::now();

    AIHoloImager::AIHoloImager imager;
    AIHoloImager::Mesh mesh = imager.Generate(input_images);

    auto elapse = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start);

    AIHoloImager::SaveMesh(mesh, output_path);

    std::cout << "Total time: " << elapse.count() << " s\n";

    return 0;
}
