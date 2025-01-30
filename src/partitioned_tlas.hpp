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


#define IMGUI_DEFINE_MATH_OPERATORS  // ImGUI ImVec maths


#include "imgui/imgui_axis.hpp"
#include "imgui/imgui_camera_widget.h"
#include "imgui/imgui_helper.h"
#include "nvh/primitives.hpp"
#include "nvvk/buffers_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/raytraceKHR_vk.hpp"
#include "nvvk/sbtwrapper_vk.hpp"
#include "nvvk/shaders_vk.hpp"
#ifdef USE_VMA
#include "nvvkhl/alloc_vma.hpp"
#else
#include "vk_mem_alloc.h"
#include "nvvk/resourceallocator_vk.hpp"
#include "nvvk/context_vk.hpp"
#include "nvvk/memallocator_dma_vk.hpp"
#endif
#include "nvvkhl/element_benchmark_parameters.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/element_gui.hpp"

#include "nvvkhl/gbuffer.hpp"
#include "nvvkhl/pipeline_container.hpp"
#include "nvvkhl/shaders/dh_sky.h"

#include "shaders/dh_bindings.h"

#include "nvvk/shadermodulemanager_vk.hpp"

#include <unordered_map>

// Generate a small domino layout by default, more practical for debugging and smaller GPUs
//#define SMALL_SCENE

// Workaround for sporadic crash issue in preview driver 570.12. Comment out to disable once fixed.
//#define MAX_INSTANCES_PER_PARTITION 1024

// Rebuild the full PTLAS 10000 times at a startup to stress test the rebuild mechanism
//#define REBUILD_STRESS_TEST

// Max depth of recursive raytracing
#define MAXRAYRECURSIONDEPTH 10


namespace DH {
using namespace glm;
#include "shaders/device_host.h"  // Shared between host and device
}  // namespace DH

#ifdef USE_NVVK_INSPECTOR
#include "nvvkhl/element_inspector.hpp"
extern std::shared_ptr<nvvkhl::ElementInspector> g_elementInspector;

enum InspectedBuffers
{
  eState0,
  eState1,
  eStateOriginal,
  eInstanceWrite,
  eInstanceWriteOriginal,
  ePartitionWrite,
  ePartitionWriteOriginal,
  eOps,
  eInstanceIndices,
  eInstanceIndicesOriginal,
  ePartitionState,
  eInspectedBufferCount
};


#endif

#include "nvvk/acceleration_structures.hpp"
#include "glm/gtx/spline.hpp"
#include "nvvk/compute_vk.hpp"


#include "imgui/imgui_orient.h"
#include "nvvkhl/element_profiler.hpp"


#include "vk_nv_partitioned_acc.h"
#include "partitioned_acceleration_structures.hpp"


//////////////////////////////////////////////////////////////////////////
/// </summary> Ray trace multiple primitives
class PartitionedTlasSample : public nvvkhl::IAppElement
{
public:
  PartitionedTlasSample()           = default;
  ~PartitionedTlasSample() override = default;

  // partitioned_tlas.cpp
  //----------------------
  // Application setup
  void onAttach(nvvkhl::Application* app) override;
  // Application destruction
  void onDetach() override;
  // Main render function
  void onRender(VkCommandBuffer cmd) override;

  // partitioned_tlas_ui.cpp
  //----------------------
  // Reset the GBuffer and image bindings
  void onResize(uint32_t width, uint32_t height) override;
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
  void writeRtDesc();


  nvvkhl::Application*             m_app = nullptr;
  std::unique_ptr<nvvk::DebugUtil> m_dutil;
#ifdef USE_VMA
  std::unique_ptr<nvvkhl::AllocVma> m_alloc;
#else
  std::unique_ptr<nvvk::ResourceAllocatorDma> m_alloc;
#endif
  std::unique_ptr<nvvk::DescriptorSetContainer> m_rtSet;  // Descriptor set

  VkFormat                            m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;  // Color format of the image
  VkDevice                            m_device      = VK_NULL_HANDLE;
  std::unique_ptr<nvvkhl::GBuffer>    m_gBuffers;  // G-Buffers
  nvvkhl_shaders::SimpleSkyParameters m_skyParams{};

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

  std::vector<nvvk::AccelKHR> m_blas;  // Bottom-level AS
  nvvk::AccelKHR              m_tlas;  // Top-level AS

  // Data and setting
  struct Material
  {
    glm::vec4 color{1.F};
  };

  // Uniques meshes
  std::vector<nvh::PrimitiveMesh> m_meshes;
  // Scene objects
  std::vector<nvh::Node> m_nodes;
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
  DH::PushConstant m_pushConst{};  // Information sent to the shader

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  nvvk::SBTWrapper          m_sbt;     // Shading binding table wrapper
  nvvkhl::PipelineContainer m_rtPipe;  // Hold pipelines and layout

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

  nvvk::PushComputeDispatcher<DH::AnimationShaderData, uint32_t, eAnimShaderCount> m_animDispatcher;
  bool                                                                             hasValidAnimationShaders() const;

  // Physics simulation data
  DH::AnimationShaderData m_animationShaderData{};
  nvvk::Buffer            m_originalState;  // Initial state of the dominoes
  nvvk::Buffer            m_state[2];       // Double-buffered current state
  nvvk::Buffer            m_globalState;    // Global simulation info, also used to keep track of the number of updates
  nvvk::Buffer            m_globalStateHost;  // Host-side simulation info for UI display
  nvvk::Buffer m_partitionState;  // Buffer keeping track of which partitions are currently touched, for illustration only
  bool m_step = false;            // Run the physics only on the next frame
  bool m_run  = false;            // Run the physics continuously

  float m_simulationSpeed = 4.f;  // Simulation speed, better leave it at that value
  float m_timeStep = 0.01f;  // Time step per simulation step, better fixed to make the simulation accuracy independent from the frame rate


#ifdef SMALL_SCENE
  int32_t m_partitionCountPerAxis = 4;
  float   m_sceneSize             = 10.f;
#else
  int32_t m_partitionCountPerAxis = 50;
  float   m_sceneSize             = 1000.f;
#endif

  // Raytracing pipeline components
  VkRayTracingPipelineCreateInfoKHR m_rayPipelineInfo{};
  enum StageIndices
  {
    eRaygen,
    eMiss,
    eClosestHit,
    eShaderGroupCount
  };

  std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> m_stages{};

  std::array<VkRayTracingShaderGroupCreateInfoKHR, eShaderGroupCount> m_shaderGroups{};


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
  nvvk::PushComputeDispatcher<DH::CompositingShaderData, uint32_t, 1> m_compositingDispatcher;
  bool                                                                m_invalidShaderNotified = false;

  // Current domino count
  uint32_t m_dominoCount{};
  // Anticipated domino count for the current regeneration parameters
  uint32_t m_generateDominoCount{};
  // Number of dominoes to automatically topple when animation starts
  int32_t m_targetToppleSeedCount{-1};

  bool m_scenePropertiesChanged{false};

  nvvk::ShaderModuleManager m_shaderManager;

  inline uint32_t partitionIndexFromPosition(const glm::vec3& position) const
  {

    uint32_t indexX = uint32_t(((position.x + m_sceneSize) * float(m_partitionCountPerAxis)) / (2.f * m_sceneSize));
    uint32_t indexY = uint32_t(((position.z + m_sceneSize) * float(m_partitionCountPerAxis)) / (2.f * m_sceneSize));

    indexX = std::min(uint32_t(m_partitionCountPerAxis) - 1, indexX);
    indexY = std::min(uint32_t(m_partitionCountPerAxis) - 1, indexY);

    // The partitions are defined as a simple ground-aligned grid
    return indexX + m_partitionCountPerAxis * indexY;
  }

public:
  // Profiler, set from outside the app class
  std::shared_ptr<nvvk::ProfilerVK> m_profilerVK;
};
