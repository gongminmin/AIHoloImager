# Data Flow in AIHoloImager

This is the overall data flow graph.

![Data flow](Img/DataFlow.png)

1. Input Images and Structure from Motion (SfM) Module:
    * The journey begins with input images, which are fed into the **Structure from Motion** module (implemented using [openMVG](https://github.com/openMVG/openMVG)). This module performs several critical tasks:
      * Generates a feature point cloud representing the scene.
      * Determines camera poses for each image.
      * Produces undistorted versions of the input images.
    * Additionally, a **Mask Generator** (implemented with [rembg](https://github.com/danielgatis/rembg)) processes the undistorted images, effectively separating foreground objects from the background. A **Delighter** (implemented with [Intrinsic](https://github.com/compphoto/Intrinsic) remove the lighting in images.

2. Reconstruct Mesh Module:
    * All the processed data--feature point cloud, camera poses, undistorted images, mask images, and delighted images--is sent to the **Mesh Reconstruction** module (implemented using [openMVS](https://github.com/cdcseacave/openMVS)). Here, a dense point cloud and the transform between it and the SfM space are generated.

3. AI Mesh Generator:
    * All the processed data are further handled by the **AI Mesh Generator** module (implemented using [TRELLIS](https://github.com/Microsoft/TRELLIS)). This AI-powered step refines the 3D mesh, enhancing its completeness.

4. Post Processing:
    * Finally, in the **Post Processing** module:
      * Project the texture to the 3D mesh.
      * Transform the mesh to a suitable pose.

In summary, AIHoloImager combines traditional and AI-driven techniques to reconstruct a comprehensive 3D mesh from a limited set of input photos.
