/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 Class definition for the Partitioned TLAS application

*/

//#define USE_VMA
#pragma once

#include "volk.h"
#define IMGUI_DEFINE_MATH_OPERATORS  // ImGUI ImVec maths
#include <vulkan/vulkan_core.h>

#include "glm/gtx/spline.hpp"
#include "nvvk/profiler_vk.hpp"

#include "partitioned_acceleration_structures.hpp"


#include "nvutils/primitives.hpp"
#include "nvvk/acceleration_structures.hpp"
#include "nvvk/descriptors.hpp"
#include "nvvk/gbuffers.hpp"
#include "nvvk/sampler_pool.hpp"
#include "nvvk/sbt_generator.hpp"
#include "nvvkglsl/glsl.hpp"
#include "nvutils/camera_manipulator.hpp"


#include <unordered_map>

// Generate a small domino layout by default, more practical for debugging and smaller GPUs
//#define SMALL_SCENE

// Workaround for sporadic crash issue in preview driver 570.12. Comment out to disable once fixed.
//#define MAX_INSTANCES_PER_PARTITION 1024

// Rebuild the full PTLAS 10000 times at a startup to stress test the rebuild mechanism
//#define REBUILD_STRESS_TEST

// Max depth of recursive raytracing
#define MAXRAYRECURSIONDEPTH 10


#include "shaders/shaderio.h"  // Shared between host and device

#ifdef USE_NVVK_INSPECTOR
#include "nvapp/elem_inspector.hpp"
extern std::shared_ptr<nvapp::ElementInspector> g_elementInspector;

enum InspectedBuffers
{
  eState0,
  eState1,
  eStateOriginal,
  eInstanceWrite,
  eInstanceWriteOriginal,
  ePartitionWriteOriginal,
  eOps,
  ePartitionState,
  eInspectedBufferCount
};


#endif
#include "nvshaders_host/sky.hpp"
#include "nvapp/application.hpp"

#include "nvvk/acceleration_structures.hpp"
#include "glm/gtx/spline.hpp"

#include "partitioned_acceleration_structures.hpp"


extern std::shared_ptr<nvutils::CameraManipulator> g_cameraManipulator;

//////////////////////////////////////////////////////////////////////////
/// </summary> Ray trace multiple primitives
class PartitionedTlasSample : public nvapp::IAppElement
{
public:
  PartitionedTlasSample()           = default;
  ~PartitionedTlasSample() override = default;

  VkCommandBuffer createTempCmdBuffer();
  void            submitAndWaitTempCmdBuffer(VkCommandBuffer cmd);
  // partitioned_tlas.cpp
  //----------------------
  // Application setup
  void onAttach(nvapp::Application* app) override;
  // Application destruction
  void onDetach() override;
  // Main render function
  void onRender(VkCommandBuffer cmd) override;

  // partitioned_tlas_ui.cpp
  //----------------------
  // Reset the GBuffer and image bindings
  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override;

  void writeCompositingDescriptors();

  // Render the app GUI
  void onUIRender() override;


private:
  // partitioned_tlas.cpp
  //----------------------
  // Destroy the resources of the app. If simpleReset is false, also destroy auxiliary objects such as the GPU memory allocator
  void destroyResources(bool simpleReset = false);
  // Reset the scene without destroying the app
  void resetScene();


  // partitioned_tlas_scene.cpp
  //----------------------------
  // Generate samples on the curve for domino layout
  std::vector<glm::vec3> generateControlPoints();
  // Compute the number of dominoes on the board
  uint32_t computeDominoCount();  //
  // Setup the domino board
  void createScene(bool resetCamera = true);
  // Create the buffers holding the scene data
  void createVkBuffers();
  // Initialize data for the physics simulation
  void createAnimationData();

  // partitioned_tlas_acceleration_structure.cpp
  //---------------------------------------------
  // Create all bottom level acceleration structures (BLAS)
  void createBottomLevelAS();

  // Create the regular top-level acceleration structure
  void createTopLevelAS();
  // Create the partitioned top-level acceleration structure
  void createPartitionedTopLevelAS();


  // partitioned_tlas_pipelines.cpp
  //---------------------------------------------
  // Create the objects related to the raytracing pipeline
  void createRtxPipeline();
  // Compile the raytracing shaders
  void recompilePipeline();
  // Compile the animation and post-processing shaders
  void recompileAuxShaders();
  // Write the descriptor set for raytracing
  void writeRaytracingDescriptors();


  nvapp::Application* m_app = nullptr;

  nvvk::ResourceAllocator m_alloc{};            // Device memory allocator
  nvvk::StagingUploader   m_stagingUploader{};  // Staging uploader for temporary buffers

  nvvk::ProfilerGpuTimer     m_profilerVK;          // GPU profiler
  nvvkglsl::GlslCompiler     m_glslCompiler{};      // GLSL compiler for on-the-fly shader compilation
  nvvk::SamplerPool          m_samplerPool{};       // The sampler pool, used to create a sampler for the texture
  nvutils::ProfilerTimeline* m_profilerTimeline{};  // Main timeline for the profiler


  VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;

  VkFormat m_colorFormat = VK_FORMAT_R32G32B32A32_SFLOAT;  // Color format of the image
  VkDevice m_device      = VK_NULL_HANDLE;

  nvvk::GBuffer                 m_gBuffers;  // G-Buffers
  shaderio::SkySimpleParameters m_skyParams{};

  // Resources
  struct PrimitiveMeshVk
  {
    nvvk::Buffer vertices;
    nvvk::Buffer indices;
  };
  std::vector<PrimitiveMeshVk> m_bMeshes;  // Each primitive holds a buffer of vertices and indices
  nvvk::Buffer                 m_bFrameInfo;
  nvvk::Buffer                 m_bInstInfoBuffer;
  nvvk::Buffer                 m_bMaterials;
  nvvk::Buffer                 m_bSkyParams;

  std::vector<nvvk::AccelerationStructure> m_blas;  // Bottom-level AS
  nvvk::AccelerationStructure              m_tlas;  // Top-level AS

  // Data and setting
  struct Material
  {
    glm::vec4 color{1.F};
  };

  // Uniques meshes
  std::vector<nvutils::PrimitiveMesh> m_meshes;
  // Scene objects
  std::vector<nvutils::Node> m_nodes;
  // Unique materials
  std::vector<Material> m_materials;

  // Height of the dominoes
  const float m_dominoHeight{1.f};
  // Spacing of the dominoes, as a factor of m_dominoHeight
  const float m_dominoSpacing{0.8f};

  // Number of movable objects ( = number of dominoes)
  uint32_t m_dynamicObjectCount{};
  // Number of static objects
  uint32_t m_staticObjectCount{};


  // Pipeline
  shaderio::PushConstant m_pushConst{};  // Information sent to the shader

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  nvvk::SBTGenerator m_sbt;        // Shading binding table wrapper
  nvvk::Buffer       m_sbtBuffer;  // Buffer holding the SBT data

  VkPipeline           m_raytracingPipeline;                        // Main rendering pipeline
  VkPipelineLayout     m_raytracingPipelineLayout{VK_NULL_HANDLE};  // Pipeline layout for the raytracing pipeline
  nvvk::DescriptorPack m_raytracingDescriptorPack{};  // Descriptor set containing the raytracing resources

  // Raytracing pipeline components
  VkRayTracingPipelineCreateInfoKHR m_raytracingPipelineInfo{};
  enum StageIndices
  {
    eRaygen,
    eMiss,
    eClosestHit,
    eShaderGroupCount
  };

  std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> m_raytracingStages{};


  // Regular TLAS buffers and build data
  nvvk::Buffer                         m_tlasScratchBuffer;
  nvvk::Buffer                         m_instancesBuffer;
  nvvk::AccelerationStructureBuildData m_tlasBuildData;

  // Partitioned TLAS helper and buffers
  nvvk::PartitionedAccelerationStructure m_ptlas;
  // Per-instance partition ID
  std::vector<uint32_t> m_partitionIndexPerNode;

  // Buffer of VkPartitionedAccelerationStructureWriteInstanceDataNV that only contain the dominoes that actually move at the current frame
  nvvk::Buffer m_partitionedTlasInstanceWriteDynamic;


  // Animation stages
  enum AnimationShader
  {
    eAnimInit            = 0,  // Initialization of the domino states and PTLAS buffers
    eAnimPhysics         = 1,  // Physics simulation
    eAnimUpdateInstances = 2,  // Update of the PTLAS instance structures
    eAnimShaderCount
  };


  std::array<shaderc::SpvCompilationResult, eAnimShaderCount> m_animationShaders;
  VkPipelineLayout m_animationPipelineLayout{VK_NULL_HANDLE};  // Pipeline layout for the animation shaders
  std::array<shaderc::SpvCompilationResult, eAnimShaderCount> m_animationShaderModules;
  std::array<VkPipeline, eAnimShaderCount>                    m_animationPipelines;

  bool hasValidAnimationShaders() const;

  // Physics simulation data
  shaderio::AnimationShaderData m_animationShaderData{};
  nvvk::Buffer                  m_originalState;  // Initial state of the dominoes
  nvvk::Buffer                  m_state[2];       // Double-buffered current state
  nvvk::Buffer m_globalState;      // Global simulation info, also used to keep track of the number of updates
  nvvk::Buffer m_globalStateHost;  // Host-side simulation info for UI display
  nvvk::Buffer m_partitionState;  // Buffer keeping track of which partitions are currently touched, for illustration only
  bool m_step = false;            // Run the physics only on the next frame
  bool m_run  = true;            // Run the physics continuously

  float m_simulationSpeed = 4.f;  // Simulation speed, better leave it at that value
  float m_timeStep = 0.01f;  // Time step per simulation step, better fixed to make the simulation accuracy independent from the frame rate


#ifdef SMALL_SCENE
  int32_t m_partitionCountPerAxis = 4;
  float   m_sceneSize             = 10.f;
#else
  int32_t m_partitionCountPerAxis = 50;
  float   m_sceneSize             = 1000.f;
#endif


  uint32_t m_sizeMultiplier = 2;  // Poor man's antialiasing by doubling the render size
  struct Statistics
  {
    size_t ptlasAccelerationStructureSize{0ull};
    size_t ptlasBuildScratchSize{0ull};

    size_t tlasAccelerationStructureSize{0ull};
    size_t tlasBuildScratchSize{0ull};
  };
  Statistics m_stats{};

  // If true, use refit rather than rebuild for regular TLAS
  bool m_tlasRefit = true;

  // Final compositing/cell shading
  shaderc::SpvCompilationResult m_compositingShader;
  VkPipeline                    m_compositingPipeline       = VK_NULL_HANDLE;
  VkPipelineLayout              m_compositingPipelineLayout = {VK_NULL_HANDLE};
  nvvk::DescriptorPack          m_compositingDescriptorPack{};  // Descriptor set


  bool m_invalidShaderNotified = false;

  // Current domino count
  uint32_t m_dominoCount{};
  // Anticipated domino count for the current regeneration parameters
  uint32_t m_generateDominoCount{};
  // Number of dominoes to automatically topple when animation starts
  int32_t m_targetToppleSeedCount{2000};

  bool m_scenePropertiesChanged{false};


  inline uint32_t partitionIndexFromPosition(const glm::vec3& position) const
  {

    uint32_t indexX = uint32_t(((position.x + m_sceneSize) * float(m_partitionCountPerAxis)) / (2.f * m_sceneSize));
    uint32_t indexY = uint32_t(((position.z + m_sceneSize) * float(m_partitionCountPerAxis)) / (2.f * m_sceneSize));

    indexX = std::min(uint32_t(m_partitionCountPerAxis) - 1, indexX);
    indexY = std::min(uint32_t(m_partitionCountPerAxis) - 1, indexY);

    // The partitions are defined as a simple ground-aligned grid
    return indexX + m_partitionCountPerAxis * indexY;
  }
};
