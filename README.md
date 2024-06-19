# AIHoloImager

In the futuristic world of Star Trek: Voyager, the Doc is an avid fan of holograms. Armed with a holo-imager, a device akin to a camera, he captures moments in 3D splendor. These holograms come to life within the holodeck, a virtual reality environment. But how does one transform mere photos into intricate 3D structures? The show leaves this question unanswered, as it operates in the realm of 24th-century technology.

Now, let¡¯s shift our focus back to reality. In our world, we have tools like structure from motion and multi-view stereo that can generate 3D meshes from a handful of images. However, there¡¯s a catch: these methods work primarily on visible surfaces. Any regions not captured by the photos remain blank, leaving gaps in the reconstructed model. On the flip side, AI-powered techniques can create a complete 3D mesh from a single image. Yet, they rely on educated guesses for content not explicitly present in that image.

So, here¡¯s the intriguing proposition: AIHoloImage aims to combine the best of both worlds. By breaking down the reconstruction pipeline into distinct stages, we can selectively apply either AI-driven or traditional methods for each stage. The result? A relatively complete 3D mesh assembled from just a few photos or a video clip.

# Getting started

AIHoloImager currently supports only Windows. To get started, use CMake to generate the project file and build the application. Before you begin, ensure that you have the following prerequisites installed:

* [Git](http://git-scm.com/downloads). Put git into the PATH is recommended.
* [Visual Studio 2022](https://www.visualstudio.com/downloads).
* [CMake](https://www.cmake.org/download/). Version 3.20 or up. It's highly recommended to choose "Add CMake to the system PATH for all users" during installation.

# License

AIHoloImager is distributed under the terms of MIT License. See [LICENSE](LICENSE) for details.
