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

  Main file for the Partitioned TLAS sample. This sample demonstrates how to use the VK_NV_partitioned_acceleration_structure

*/

#ifdef USE_VMA
#define VMA_IMPLEMENTATION
#endif

#include "partitioned_tlas.hpp"
#include "vk_context.hpp"
#include "nvvk/extensions_vk.hpp"
#ifdef USE_NVVK_INSPECTOR
std::shared_ptr<nvvkhl::ElementInspector> g_elementInspector;
#endif

// Generic memory barrier, used for synchronization across different stages
static void memoryBarrier(VkCommandBuffer cmd)
{
  VkMemoryBarrier      mb{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                          .srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT
                                      | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR
                                      | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                          .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT
                                      | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR
                                      | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR};
  VkPipelineStageFlags srcDstStage{VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
  vkCmdPipelineBarrier(cmd, srcDstStage, srcDstStage, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

// App initialization
void PartitionedTlasSample::onAttach(nvvkhl::Application* app)
{
  nvh::ScopedTimer st(__FUNCTION__);

  m_app    = app;
  m_device = m_app->getDevice();

  m_app->setVsync(true);

  m_shaderManager.init(m_device, 1, 3);
  {
    std::vector<std::string> shaderSearchPaths;
    std::string              path = NVPSystem::exePath();
    shaderSearchPaths.push_back(NVPSystem::exePath());
    shaderSearchPaths.push_back(NVPSystem::exePath() + "shaders");
    shaderSearchPaths.push_back(std::string("GLSL_" PROJECT_NAME));
    shaderSearchPaths.push_back(NVPSystem::exePath() + std::string("GLSL_" PROJECT_NAME));
    shaderSearchPaths.push_back(NVPSystem::exePath() + std::string(PROJECT_RELDIRECTORY));
    shaderSearchPaths.push_back(NVPSystem::exePath() + std::string(PROJECT_RELDIRECTORY) + "shaders");
    shaderSearchPaths.push_back(NVPSystem::exePath() + std::string("../") + "shaders");
    shaderSearchPaths.push_back(NVPSystem::exePath() + std::string(PROJECT_NVPRO_CORE_RELDIRECTORY));

    m_shaderManager.m_filetype        = nvh::ShaderFileManager::FILETYPE_GLSL;
    m_shaderManager.m_keepModuleSPIRV = true;
    for(const auto& it : shaderSearchPaths)
    {
      m_shaderManager.addDirectory(it);
    }
  }


  m_dutil = std::make_unique<nvvk::DebugUtil>(m_device);  // Debug utility

#ifdef USE_VMA
  m_alloc = std::make_unique<nvvkhl::AllocVma>(m_app->getContext().get());  // Allocator
#else
  m_alloc = std::make_unique<nvvk::ResourceAllocatorDma>(m_app->getDevice(), m_app->getPhysicalDevice());  // Allocator
#endif
  m_rtSet = std::make_unique<nvvk::DescriptorSetContainer>(m_device);

  // Requesting ray tracing properties
  VkPhysicalDeviceProperties2 prop2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &m_rtProperties};
  vkGetPhysicalDeviceProperties2(m_app->getPhysicalDevice(), &prop2);



#ifdef USE_NVVK_INSPECTOR
  nvvkhl::ElementInspector::InitInfo inspectorInitInfo{};

  inspectorInitInfo.allocator                = m_alloc.get();
  inspectorInitInfo.bufferCount              = eInspectedBufferCount;
  inspectorInitInfo.computeCount             = 0;
  inspectorInitInfo.customCount              = 0;
  inspectorInitInfo.device                   = m_device;
  inspectorInitInfo.fragmentCount            = 0;
  inspectorInitInfo.graphicsQueueFamilyIndex = m_app->getQueueGCT().familyIndex;
  inspectorInitInfo.imageCount               = 0;
  g_elementInspector->init(inspectorInitInfo);
#endif


  // Create utilities to create BLAS/TLAS and the Shading Binding Table (SBT)
  const uint32_t queueIndexGCT = m_app->getQueue(0).familyIndex;
  m_sbt.setup(m_device, queueIndexGCT, m_alloc.get(), m_rtProperties);

  // Create resources
  createScene();
  createVkBuffers();
  createBottomLevelAS();
  createTopLevelAS();
  createRtxPipeline();
  createAnimationData();
  createPartitionedTopLevelAS();
}

void PartitionedTlasSample::onDetach()
{
  vkDeviceWaitIdle(m_device);
  destroyResources();
#ifdef USE_NVVK_INSPECTOR
  g_elementInspector->deinit();
#endif
}

// Reset after shader reloading, change of scene contents etc
void PartitionedTlasSample::resetScene()
{
  vkDeviceWaitIdle(m_device);
  destroyResources(true);

  m_nodes.clear();
  m_meshes.clear();
  m_materials.clear();

  m_rtSet->init(m_device);
  const uint32_t queueIndexGCT = m_app->getQueue(0).familyIndex;
  m_sbt.setup(m_device, queueIndexGCT, m_alloc.get(), m_rtProperties);

  m_tlasBuildData = {};
  m_run           = false;
  // Create resources
  createScene(false);
  createVkBuffers();
  createBottomLevelAS();
  createTopLevelAS();
  createRtxPipeline();
  createAnimationData();
  createPartitionedTopLevelAS();
  writeRtDesc();

  vkDeviceWaitIdle(m_device);
}


bool PartitionedTlasSample::hasValidAnimationShaders() const
{
  for(size_t i = 0; i < eAnimShaderCount; i++)
  {
    if(m_animDispatcher.getPipeline(int32_t(i)) == VK_NULL_HANDLE)
      return false;
  }
  return true;
}

void PartitionedTlasSample::onRender(VkCommandBuffer cmd)
{
  if(m_rtPipe.plines[0] == VK_NULL_HANDLE)
  {
    if(!m_invalidShaderNotified)
    {
      LOGE("Invalid raytracing pipeline\n");
      m_invalidShaderNotified = true;
    }
    return;
  }

  const nvvk::DebugUtil::ScopedCmdLabel sdbg = m_dutil->DBG_SCOPE(cmd);

  // Camera matrices
  glm::mat4 proj = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), CameraManip.getAspectRatio(),
                                         CameraManip.getClipPlanes().x, CameraManip.getClipPlanes().y);
  proj[1][1] *= -1;  // Vulkan has it inverted

  // Update uniform buffers
  DH::FrameInfo finfo{.projInv = glm::inverse(proj), .viewInv = glm::inverse(CameraManip.getMatrix())};
  vkCmdUpdateBuffer(cmd, m_bFrameInfo.buffer, 0, sizeof(DH::FrameInfo), &finfo);  // Update FrameInfo
  vkCmdUpdateBuffer(cmd, m_bSkyParams.buffer, 0, sizeof(nvvkhl_shaders::SimpleSkyParameters), &m_skyParams);  // Update the sky
  memoryBarrier(cmd);  // Make sure the data has moved to device before rendering

  m_animationShaderData.eyePosition = CameraManip.getEye();
  m_animationShaderData.partitionCount = m_partitionCountPerAxis * m_partitionCountPerAxis + 1;  // +1 for the global partition

  if(m_step || m_run)
  {
    if(hasValidAnimationShaders())
    {
      uint32_t zero = 0u;
      vkCmdUpdateBuffer(cmd, m_globalState.buffer, offsetof(DH::AnimationGlobalState, instanceUpdateCount), sizeof(uint32_t), &zero);

      if(m_animationShaderData.ptlasActive != 0)
      {
        // Reset the instance write operation: the number of instances to update will be written by the instance update shader
        VkBuildPartitionedAccelerationStructureIndirectCommandNV ptlasOps;
        ptlasOps.argCount              = 0;
        ptlasOps.opType                = VK_PARTITIONED_ACCELERATION_STRUCTURE_OP_TYPE_WRITE_INSTANCE_NV;
        ptlasOps.argData.startAddress  = m_partitionedTlasInstanceWriteDynamic.address;
        ptlasOps.argData.strideInBytes = sizeof(VkPartitionedAccelerationStructureWriteInstanceDataNV);

        vkCmdUpdateBuffer(cmd, m_ptlas.getBuffers().operationsInfo.buffer, 0,
                          sizeof(VkBuildPartitionedAccelerationStructureIndirectCommandNV), &ptlasOps);

        // The next PTLAS build will contain one operation to update the instances
        uint32_t srcCount = 1;
        vkCmdUpdateBuffer(cmd, m_ptlas.getBuffers().operationsCount.buffer, 0, sizeof(uint32_t), &srcCount);

        // Update the buffer references for use in the animation shaders
        // PTLAS update operation definitions
        m_animationShaderData.ptlasOperations = m_ptlas.getBuffers().operationsInfo.address;
        // Instance write operations for regular TLAS
        m_animationShaderData.instanceWriteInfo = m_partitionedTlasInstanceWriteDynamic.address;
      }

      memoryBarrier(cmd);

      // If the dominoes need to be set to their original state, run the animation initialization shader and rewrite the
      // instance indices for each partition
      if(m_animationShaderData.resetToOriginal != 0)
      {
        m_animDispatcher.dispatchThreads(
            cmd, std::max(m_animationShaderData.totalObjectCount, uint32_t(m_animationShaderData.partitionCount)), &m_animationShaderData,
            nvvk::DispatcherBarrier::eCompute, nvvk::DispatcherBarrier::eTransfer, ANIMATION_SHADER_BLOCK_SIZE, eAnimInit);
        m_animationShaderData.resetToOriginal = 0;
        m_step                                = false;
        memoryBarrier(cmd);
      }
      else
      {
        // Run the physics
        {
          // Number of subframes for the physics simulation
          const uint32_t subFrames            = 5;
          auto           timerSection         = m_profilerVK->timeRecurring("Physics", cmd);
          m_animationShaderData.timeStep      = m_timeStep * m_simulationSpeed / float(subFrames);
          m_animationShaderData.subframeCount = uint32_t(subFrames);


          // Run the actual physics sim
          m_animDispatcher.dispatchThreads(cmd, uint32_t(m_animationShaderData.dynamicObjectCount),
                                           &m_animationShaderData, nvvk::DispatcherBarrier::eCompute,
                                           nvvk::DispatcherBarrier::eNone, ANIMATION_SHADER_BLOCK_SIZE, eAnimPhysics);
        }
        memoryBarrier(cmd);
        if(m_animationShaderData.ptlasActive != 0)
        {
          // If PTLAS is active, run the instance update and partition update shaders
          m_animDispatcher.dispatchThreads(cmd, uint32_t(m_animationShaderData.dynamicObjectCount), &m_animationShaderData,
                                           nvvk::DispatcherBarrier::eCompute, nvvk::DispatcherBarrier::eNone,
                                           ANIMATION_SHADER_BLOCK_SIZE, eAnimUpdateInstances);

          memoryBarrier(cmd);
        }

        m_animationShaderData.currentStateIndex = (m_animationShaderData.currentStateIndex + 1) % 2;


        m_step = false;
      }


#ifdef USE_NVVK_INSPECTOR
#ifndef NVVK_INSPECTOR_OPS_ONLY
      g_elementInspector->inspectBuffer(cmd, eState0);
      g_elementInspector->inspectBuffer(cmd, eState1);
#endif
      if(m_animationShaderData.ptlasActive != 0)
      {
        g_elementInspector->inspectBuffer(cmd, eOps);
#ifndef NVVK_INSPECTOR_OPS_ONLY
        g_elementInspector->inspectBuffer(cmd, ePartitionState);
        g_elementInspector->inspectBuffer(cmd, ePartitionWrite);
        g_elementInspector->inspectBuffer(cmd, ePartitionWriteOriginal);
        g_elementInspector->inspectBuffer(cmd, eInstanceWrite);
        g_elementInspector->inspectBuffer(cmd, eInstanceWriteOriginal);
        g_elementInspector->inspectBuffer(cmd, eInstanceIndices);
        g_elementInspector->inspectBuffer(cmd, eInstanceIndicesOriginal);

        g_elementInspector->inspectBuffer(cmd, eStateOriginal);
#endif
      }
      memoryBarrier(cmd);
#endif


      // Read back the global animation state to display update information in the UI
      {
        VkBufferCopy region;
        region.dstOffset = 0;
        region.srcOffset = 0;
        region.size      = sizeof(DH::AnimationGlobalState);
        vkCmdCopyBuffer(cmd, m_globalState.buffer, m_globalStateHost.buffer, 1, &region);
      }

      // Perform the update of the (P)TLAS
      {
        auto timerSection = m_profilerVK->timeRecurring("TLAS Update", cmd);
        if(m_animationShaderData.ptlasActive != 0)
        {
          // PTLAS update
          m_ptlas.buildAccelerationStructure(cmd, true);
        }
        else
        {
          if(m_tlasRefit)
          {
            // TLAS refit only
            m_tlasBuildData.cmdUpdateAccelerationStructure(cmd, m_tlas.accel, m_tlasScratchBuffer.address);
          }
          else
          {
            // Full TLAS rebuild
            m_tlasBuildData.cmdBuildAccelerationStructure(cmd, m_tlas.accel, m_tlasScratchBuffer.address);
          }
        }

        memoryBarrier(cmd);
      }
    }
    else
    {
      if(!m_invalidShaderNotified)
      {
        LOGE("Invalid animation shader\n");
        m_invalidShaderNotified = true;
      }
    }
  }


  {
    auto timerSection = m_profilerVK->timeRecurring("Render", cmd);
    // Ray trace
    std::vector<VkDescriptorSet> desc_sets{m_rtSet->getSet()};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipe.plines[0]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipe.layout, 0, (uint32_t)desc_sets.size(),
                            desc_sets.data(), 0, nullptr);

    m_pushConst.animationShaderData = m_animationShaderData;

    vkCmdPushConstants(cmd, m_rtPipe.layout, VK_SHADER_STAGE_ALL, 0, sizeof(DH::PushConstant), &m_pushConst);

    const std::array<VkStridedDeviceAddressRegionKHR, 4>& bindingTables = m_sbt.getRegions();
    const VkExtent2D&                                     size          = m_app->getViewportSize();
    vkCmdTraceRaysKHR(cmd, &bindingTables[0], &bindingTables[1], &bindingTables[2], &bindingTables[3],
                      size.width * m_sizeMultiplier, size.height * m_sizeMultiplier, 1);
  }

  // Post-processing for AO filtering, toon shading etc
  {
    auto                      timerSection = m_profilerVK->timeRecurring("Compositing", cmd);
    const VkExtent2D&         size         = m_app->getViewportSize();
    DH::CompositingShaderData compositingShaderData;
    compositingShaderData.windowSize = 3;
    compositingShaderData.frameIndex = m_animationShaderData.frameIndex;
    if(m_compositingDispatcher.getPipeline(0) != VK_NULL_HANDLE)
    {
      for(uint32_t i = 0; i < 4; i++)
      {
        compositingShaderData.passId = i;
        m_compositingDispatcher.dispatchThreads(cmd, size.width * size.height * m_sizeMultiplier * m_sizeMultiplier,
                                                &compositingShaderData);
      }
    }
    else
    {
      if(!m_invalidShaderNotified)
      {
        LOGE("Invalid compositing shader\n");
        m_invalidShaderNotified = true;
      }
    }
  }
  m_animationShaderData.frameIndex++;
}


void PartitionedTlasSample::destroyResources(bool simpleReset)
{
  for(PrimitiveMeshVk& m : m_bMeshes)
  {
    m_alloc->destroy(m.vertices);
    m_alloc->destroy(m.indices);
  }
  m_bMeshes.clear();

  m_alloc->destroy(m_bFrameInfo);
  m_alloc->destroy(m_bInstInfoBuffer);
  m_alloc->destroy(m_bMaterials);
  m_alloc->destroy(m_bSkyParams);

  m_rtSet->deinit();

  if(!simpleReset)
  {
    m_gBuffers.reset();
  }

  m_rtPipe.destroy(m_device);
  m_sbt.destroy();
  m_compositingDispatcher.deinit();
  m_animDispatcher.deinit();

  for(auto& b : m_blas)
    m_alloc->destroy(b);
  m_blas.clear();
  m_alloc->destroy(m_tlas);

  nvvk::PartitionedAccelerationStructure::Buffers ptlas = m_ptlas.getBuffers();

  m_alloc->destroy(ptlas.accelerationStructure);
  m_alloc->destroy(ptlas.buildScratch);
  m_alloc->destroy(ptlas.instanceUpdateInfo);
  m_alloc->destroy(ptlas.instanceWriteInfo);
  m_alloc->destroy(ptlas.operationsCount);
  m_alloc->destroy(ptlas.operationsInfo);
  m_alloc->destroy(ptlas.partitionWriteInfo);
  m_alloc->destroy(ptlas.updateScratch);

  m_alloc->destroy(m_globalState);
  m_alloc->destroy(m_globalStateHost);

  m_alloc->destroy(m_originalState);
  m_alloc->destroy(m_state[0]);
  m_alloc->destroy(m_state[1]);
  m_alloc->destroy(m_partitionState);

  m_alloc->destroy(m_partitionedTlasInstanceWriteDynamic);

  m_alloc->destroy(m_instancesBuffer);

  m_alloc->destroy(m_tlasScratchBuffer);

  m_alloc->releaseStaging();

  m_ptlas.deinit();

  if(!simpleReset)
  {
    m_shaderManager.deinit();
  }
  if(!simpleReset)
  {
    m_alloc->deinit();
  }
}


//////////////////////////////////////////////////////////////////////////
///
//////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  try
  {
    VkContextSettings vkSetup;
    // Disable validation layers as they may pose issues with newer extensions
    vkSetup.enableValidationLayers = false;
    vkSetup.apiVersion             = VK_API_VERSION_1_4;

    nvvkhl::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    vkSetup.instanceExtensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    vkSetup.deviceExtensions.emplace_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    vkSetup.deviceExtensions.emplace_back(ExtensionFeaturePair{VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accel_feature});  // To build acceleration structures
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_pipeline_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
    vkSetup.deviceExtensions.emplace_back(ExtensionFeaturePair{VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rt_pipeline_feature});  // To use vkCmdTraceRaysKHR
    vkSetup.deviceExtensions.emplace_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);  // Required by ray tracing pipeline
    vkSetup.deviceExtensions.emplace_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);

    VkPhysicalDevicePartitionedAccelerationStructureFeaturesNV ptlasFeature{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PARTITIONED_ACCELERATION_STRUCTURE_FEATURES_NV, nullptr, VK_TRUE};

    vkSetup.deviceExtensions.emplace_back(ExtensionFeaturePair{VK_NV_PARTITIONED_ACCELERATION_STRUCTURE_EXTENSION_NAME, &ptlasFeature,
                                                               VK_NV_PARTITIONED_ACCELERATION_STRUCTURE_SPEC_VERSION, true});

    // Request the creation of all needed queues
    vkSetup.queues = {VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT | VK_QUEUE_COMPUTE_BIT};

    // Create Vulkan context
    auto vkContext = std::make_unique<VkContext>(vkSetup);
    if(!vkContext->isValid())
      std::exit(0);

    // Loading Vulkan extension pointers
    load_VK_EXTENSIONS(vkContext->getInstance(), vkGetInstanceProcAddr, vkContext->getDevice(), vkGetDeviceProcAddr);


    nvvkhl::ApplicationCreateInfo spec;
    spec.name           = PROJECT_NAME;
    spec.vSync          = false;
    spec.instance       = vkContext->getInstance();
    spec.device         = vkContext->getDevice();
    spec.physicalDevice = vkContext->getPhysicalDevice();
    spec.queues         = vkContext->getQueueInfos();

    // Create the application
    auto app = std::make_unique<nvvkhl::Application>(spec);

    // Create the test framework
    auto test = std::make_shared<nvvkhl::ElementBenchmarkParameters>(argc, argv);

    auto partitionedTlas = std::make_shared<PartitionedTlasSample>();

    auto elementProfiler = std::make_shared<nvvkhl::ElementProfiler>(true);


    partitionedTlas->m_profilerVK = elementProfiler;

    // Add all application elements
    app->addElement(test);
    app->addElement(std::make_shared<nvvkhl::ElementCamera>());              // Camera manipulation
    app->addElement(std::make_shared<nvvkhl::ElementDefaultMenu>());         // Menu / Quit
    app->addElement(std::make_shared<nvvkhl::ElementDefaultWindowTitle>());  // Window title info
    app->addElement(elementProfiler);
#ifdef USE_NVVK_INSPECTOR
    g_elementInspector = std::make_shared<nvvkhl::ElementInspector>();
    app->addElement(g_elementInspector);
#endif

    app->addElement(partitionedTlas);

    app->run();
    app.reset();
    vkContext.reset();
    return test->errorCode();
  }
  catch(const std::exception& e)
  {

    LOGE("Uncaught exception: %s\n", e.what());
    assert(!"We should never reach here under normal operation, but this "
            "prints a nicer error message in the event we do.");
    return EXIT_FAILURE;
  }
}
