// Copyright (c) 2024 Minmin Gong
//

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <format>
#include <iostream>

#ifndef _DEBUG
    #define CXXOPTS_NO_RTTI
#endif
#include <cxxopts.hpp>

#include "AIHoloImager/AIHoloImager.hpp"

int main(int argc, char* argv[])
{
    cxxopts::Options options("AIHoloImager", "AIHoloImager: AI generated mesh from photos.");
    // clang-format off
    options.add_options()
        ("H,help", "Produce help message.")
        ("I,input-path", "The directory that contains the input image sequence.", cxxopts::value<std::string>())
        ("O,output-path", "The path of the output mesh (\"<input-dir>/Output/Mesh.glb\" by default).", cxxopts::value<std::string>())
        ("D,device", "The computation device for inferencing (cuda or cpu; cuda by default).", cxxopts::value<std::string>())
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
        std::cout << "AIHoloImager, Version 0.4.0\n";
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

    auto device = AIHoloImager::AIHoloImager::DeviceType::Cuda;
    if (vm.count("device") > 0)
    {
        std::string device_name = vm["device"].as<std::string>();
        std::transform(
            device_name.begin(), device_name.end(), device_name.begin(), [](char c) { return static_cast<char>(std::tolower(c)); });

        if (device_name == "cpu")
        {
            device = AIHoloImager::AIHoloImager::DeviceType::Cpu;
        }
        else if (device_name == "cuda")
        {
            device = AIHoloImager::AIHoloImager::DeviceType::Cuda;
        }
        else
        {
            std::cerr << std::format("ERROR: Unsupported device {}. Use cpu or cuda.\n", device_name);
            return 1;
        }
    }

    std::filesystem::create_directories(output_path.parent_path());

    const auto tmp_dir = output_path.parent_path() / "Tmp";
    std::filesystem::create_directories(tmp_dir);

    auto start = std::chrono::high_resolution_clock::now();

    try
    {
        AIHoloImager::AIHoloImager imager(device, tmp_dir);
        const AIHoloImager::Mesh mesh = imager.Generate(input_path, 2048);
        AIHoloImager::SaveMesh(mesh, output_path);

#ifndef AIHI_KEEP_INTERMEDIATES
        std::filesystem::remove_all(tmp_dir);
#endif
    }
    catch (std::exception& ex)
    {
        std::cout << "ERROR: " << ex.what() << '\n';
    }

    auto elapse = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start);

    std::cout << "Total time: " << elapse.count() << " s\n";

    return 0;
}
