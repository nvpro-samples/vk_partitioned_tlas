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
#define USE_NSIGHT_AFTERMATH 1

#define VMA_IMPLEMENTATION

#include <fmt/format.h>
#include "partitioned_tlas.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif


#include "shaders/shaderio.h"  // Shared between host and device

#ifdef USE_NVVK_INSPECTOR
std::shared_ptr<nvapp::ElementInspector> g_elementInspector;
#endif


#include "nvaftermath/aftermath.hpp"
#include "nvapp/application.hpp"
#include "nvapp/elem_camera.hpp"
#include "nvapp/elem_default_menu.hpp"
#include "nvapp/elem_default_title.hpp"
#include "nvapp/elem_profiler.hpp"
#include "nvutils/file_operations.hpp"
#include "nvutils/logger.hpp"
#include "nvvk/context.hpp"
#include "nvvk/debug_util.hpp"
#include "nvvk/resource_allocator.hpp"
#include <nvutils/parameter_parser.hpp>
#include <nvutils/parameter_registry.hpp>
#include <nvvk/validation_settings.hpp>

nvutils::ProfilerManager                    g_profilerManager;  // #PROFILER
std::shared_ptr<nvutils::CameraManipulator> g_cameraManipulator;


// Create a temporary command buffer
VkCommandBuffer PartitionedTlasSample::createTempCmdBuffer()
{
  VkCommandBufferAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  allocateInfo.commandBufferCount = 1;
  allocateInfo.commandPool        = m_app->getCommandPool();
  allocateInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  VkCommandBuffer cmd             = nullptr;
  NVVK_CHECK(vkAllocateCommandBuffers(m_device, &allocateInfo, &cmd));

  VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  NVVK_CHECK(vkBeginCommandBuffer(cmd, &begin_info));
  return cmd;
}

// Submit the temporary command buffer
void PartitionedTlasSample::submitAndWaitTempCmdBuffer(VkCommandBuffer cmd)
{
  NVVK_CHECK(vkEndCommandBuffer(cmd));

  VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers    = &cmd;
  NVVK_CHECK(vkQueueSubmit(m_app->getQueue(0).queue, 1, &submitInfo, {}));
  NVVK_CHECK(vkQueueWaitIdle(m_app->getQueue(0).queue));
  vkFreeCommandBuffers(m_device, m_app->getCommandPool(), 1, &cmd);
}

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
void PartitionedTlasSample::onAttach(nvapp::Application* app)
{
  nvutils::ScopedTimer st(__FUNCTION__);

  m_app              = app;
  m_device           = m_app->getDevice();
  m_descriptorPool   = app->getTextureDescriptorPool();
  m_profilerTimeline = g_profilerManager.createTimeline({.name = "Primary Timeline"});


  m_app->setVsync(true);

  m_profilerVK.init(m_profilerTimeline, m_device, m_app->getPhysicalDevice(), m_app->getQueue(0).familyIndex, false);


  // Setting up the GLSL compiler
  m_glslCompiler.defaultTarget();
  m_glslCompiler.defaultOptions();
  m_glslCompiler.options().SetGenerateDebugInfo();
  m_glslCompiler.options().SetOptimizationLevel(shaderc_optimization_level_performance);

#if defined(AFTERMATH_AVAILABLE)
  // This aftermath callback is used to report the shader hash (Spirv) to the Aftermath library.
  m_glslCompiler.setCompileCallback([&](const std::filesystem::path& sourceFile, const uint32_t* spirvCode, size_t spirvSize) {
    std::span<const uint32_t> data(spirvCode, spirvSize / sizeof(uint32_t));
    AftermathCrashTracker::getInstance().addShaderBinary(data);
  });
#endif


  {
    std::vector<std::filesystem::path> shaderSearchPaths;
    std::filesystem::path              exePath = nvutils::getExecutablePath().parent_path();
    std::filesystem::path              exeName = nvutils::getExecutablePath().stem();

    shaderSearchPaths.push_back(std::filesystem::absolute(exePath / "shaders"));
    shaderSearchPaths.push_back(std::filesystem::absolute(exePath / std::filesystem::path(PROJECT_EXE_TO_SOURCE_DIRECTORY)));
    shaderSearchPaths.push_back(std::filesystem::absolute(exePath / std::filesystem::path("../") / "shaders"));

    shaderSearchPaths.push_back(std::filesystem::absolute(exePath / std::filesystem::path(PROJECT_EXE_TO_SOURCE_DIRECTORY) / "shaders"));
    shaderSearchPaths.push_back(std::filesystem::absolute(exePath / std::filesystem::path(PROJECT_EXE_TO_NVSHADERS_DIRECTORY)));
    shaderSearchPaths.push_back(std::filesystem::absolute(exePath / std::filesystem::path(PROJECT_EXE_TO_ROOT_DIRECTORY) / "common"));
    shaderSearchPaths.push_back(std::filesystem::absolute(exePath / exeName / "shaders"));
    shaderSearchPaths.push_back(std::filesystem::absolute(exePath));

    m_glslCompiler.addSearchPaths(shaderSearchPaths);
  }

  // Setting up the shader compiler callback
#if defined(AFTERMATH_AVAILABLE)
  m_glslCompiler.setCompileCallback([&](const std::filesystem::path& sourceFile, const uint32_t* spirvCode, size_t spirvSize) {
    // nvvk::dumpSpirvWithHashedName(sourceFile, spirvCode, spirvSize);  // Extra <--- Dumping the Spir-V

    auto&                     aftermath = AftermathCrashTracker::getInstance();
    std::span<const uint32_t> data(spirvCode, spirvSize / sizeof(uint32_t));
    aftermath.addShaderBinary(data);
  });
#endif

  // Initialize the VMA allocator
  m_alloc.init({
      .physicalDevice   = m_app->getPhysicalDevice(),
      .device           = m_app->getDevice(),
      .instance         = m_app->getInstance(),
      .vulkanApiVersion = VK_API_VERSION_1_4,
  });

  // Initialize the utility for uploads
  m_stagingUploader.init(&m_alloc, true);


  // Acquiring the sampler which will be used for displaying the GBuffer
  m_samplerPool.init(m_app->getDevice());
  //VkSampler linearSampler;
  //NVVK_CHECK(m_samplerPool.acquireSampler(linearSampler));
  //NVVK_DBG_NAME(linearSampler);


  // Requesting ray tracing properties
  VkPhysicalDeviceProperties2 prop2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &m_rtProperties};
  vkGetPhysicalDeviceProperties2(m_app->getPhysicalDevice(), &prop2);


#ifdef USE_NVVK_INSPECTOR
  nvapp::ElementInspector::InitInfo inspectorInitInfo{};

  inspectorInitInfo.allocator             = &m_alloc;
  inspectorInitInfo.bufferCount           = eInspectedBufferCount;
  inspectorInitInfo.computeCount          = 0;
  inspectorInitInfo.customCount           = 0;
  inspectorInitInfo.device                = m_device;
  inspectorInitInfo.fragmentCount         = 0;
  inspectorInitInfo.imageCount            = 0;
  inspectorInitInfo.queueInfo.familyIndex = m_app->getQueue(0).familyIndex;
  inspectorInitInfo.queueInfo.queue       = m_app->getQueue(0).queue;
  inspectorInitInfo.queueInfo.queueIndex  = m_app->getQueue(0).queueIndex;
  g_elementInspector->init(inspectorInitInfo);
#endif


  // Create utilities to create BLAS/TLAS and the Shading Binding Table (SBT)
  const uint32_t queueIndexGCT = m_app->getQueue(0).familyIndex;
  m_sbt.init(m_device, m_rtProperties);

  nvvk::GBufferInitInfo createInfo;
  createInfo.allocator      = &m_alloc;
  createInfo.colorFormats   = std::vector<VkFormat>({m_colorFormat, m_colorFormat, VK_FORMAT_R32_SFLOAT});
  createInfo.descriptorPool = m_descriptorPool;
  m_samplerPool.acquireSampler(createInfo.imageSampler);

  m_gBuffers.init(createInfo);

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

  m_raytracingDescriptorPack.deinit();
  const uint32_t queueIndexGCT = m_app->getQueue(0).familyIndex;
  m_sbt.init(m_device, m_rtProperties);

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
  writeRaytracingDescriptors();
  writeCompositingDescriptors();

  vkDeviceWaitIdle(m_device);
}


bool PartitionedTlasSample::hasValidAnimationShaders() const
{
  for(size_t i = 0; i < eAnimShaderCount; i++)
  {
    if(m_animationPipelines[i] == VK_NULL_HANDLE)
      return false;
  }
  return true;
}

void PartitionedTlasSample::onRender(VkCommandBuffer cmd)
{

  m_profilerTimeline->frameAdvance();

  if(m_raytracingPipeline == VK_NULL_HANDLE)
  {
    if(!m_invalidShaderNotified)
    {
      LOGE("Invalid raytracing pipeline\n");
      m_invalidShaderNotified = true;
    }
    return;
  }

  // Camera matrices
  glm::mat4 proj = glm::perspectiveRH_ZO(glm::radians(g_cameraManipulator->getFov()), g_cameraManipulator->getAspectRatio(),
                                         g_cameraManipulator->getClipPlanes().x, g_cameraManipulator->getClipPlanes().y);
  proj[1][1] *= -1;  // Vulkan has it inverted

  // Update uniform buffers
  shaderio::FrameInfo finfo{.projInv = glm::inverse(proj), .viewInv = glm::inverse(g_cameraManipulator->getViewMatrix())};
  vkCmdUpdateBuffer(cmd, m_bFrameInfo.buffer, 0, sizeof(shaderio::FrameInfo), &finfo);  // Update FrameInfo
  vkCmdUpdateBuffer(cmd, m_bSkyParams.buffer, 0, sizeof(shaderio::SkySimpleParameters), &m_skyParams);  // Update the sky
  memoryBarrier(cmd);  // Make sure the data has moved to device before rendering

  m_animationShaderData.eyePosition    = g_cameraManipulator->getEye();
  m_animationShaderData.partitionCount = m_partitionCountPerAxis * m_partitionCountPerAxis + 1;  // +1 for the global partition

  if(m_step || m_run)
  {
    if(hasValidAnimationShaders())
    {
      uint32_t zero = 0u;
      vkCmdUpdateBuffer(cmd, m_globalState.buffer, offsetof(shaderio::AnimationGlobalState, instanceUpdateCount),
                        sizeof(uint32_t), &zero);

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

        {
          auto timerSection = m_profilerVK.cmdFrameSection(cmd, "Animation init");
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_animationPipelines[eAnimInit]);
          vkCmdPushConstants(cmd, m_animationPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                             sizeof(shaderio::AnimationShaderData), &m_animationShaderData);

          uint32_t threadCount = std::max(m_animationShaderData.totalObjectCount, uint32_t(m_animationShaderData.partitionCount));

          uint32_t groupCount = uint32_t((threadCount + ANIMATION_SHADER_BLOCK_SIZE - 1) / ANIMATION_SHADER_BLOCK_SIZE);

          vkCmdDispatch(cmd, groupCount, 1, 1);

          memoryBarrier(cmd);
        }


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
          auto           timerSection         = m_profilerVK.cmdFrameSection(cmd, "Physics");
          m_animationShaderData.timeStep      = m_timeStep * m_simulationSpeed / float(subFrames);
          m_animationShaderData.subframeCount = uint32_t(subFrames);


          // Run the actual physics sim
          {
            auto timerSection = m_profilerVK.cmdFrameSection(cmd, "Animation physics");
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_animationPipelines[eAnimPhysics]);
            vkCmdPushConstants(cmd, m_animationPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(shaderio::AnimationShaderData), &m_animationShaderData);

            uint32_t threadCount = uint32_t(m_animationShaderData.dynamicObjectCount);

            uint32_t groupCount = uint32_t((threadCount + ANIMATION_SHADER_BLOCK_SIZE - 1) / ANIMATION_SHADER_BLOCK_SIZE);

            vkCmdDispatch(cmd, groupCount, 1, 1);

            memoryBarrier(cmd);
          }
        }
        memoryBarrier(cmd);
        if(m_animationShaderData.ptlasActive != 0)
        {
          // If PTLAS is active, run the instance update and partition update shaders

          {
            auto timerSection = m_profilerVK.cmdFrameSection(cmd, "Animation update instances");
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_animationPipelines[eAnimUpdateInstances]);
            vkCmdPushConstants(cmd, m_animationPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(shaderio::AnimationShaderData), &m_animationShaderData);

            uint32_t threadCount = uint32_t(m_animationShaderData.dynamicObjectCount);

            uint32_t groupCount = uint32_t((threadCount + ANIMATION_SHADER_BLOCK_SIZE - 1) / ANIMATION_SHADER_BLOCK_SIZE);

            vkCmdDispatch(cmd, groupCount, 1, 1);

            memoryBarrier(cmd);
          }


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
        g_elementInspector->inspectBuffer(cmd, eInstanceWrite);
        g_elementInspector->inspectBuffer(cmd, eInstanceWriteOriginal);

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
        region.size      = sizeof(shaderio::AnimationGlobalState);
        vkCmdCopyBuffer(cmd, m_globalState.buffer, m_globalStateHost.buffer, 1, &region);
      }

      // Perform the update of the (P)TLAS
      {
        auto timerSection = m_profilerVK.cmdFrameSection(cmd, "TLAS Update");
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
    auto timerSection = m_profilerVK.cmdFrameSection(cmd, "Render");
    // Ray trace
    std::vector<VkDescriptorSet> desc_sets{m_raytracingDescriptorPack.sets};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_raytracingPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_raytracingPipelineLayout, 0,
                            (uint32_t)desc_sets.size(), desc_sets.data(), 0, nullptr);

    m_pushConst.animationShaderData = m_animationShaderData;
    m_pushConst.tlas =
        m_animationShaderData.ptlasActive != 0 ? m_ptlas.getBuffers().accelerationStructure.address : m_tlas.address;

    vkCmdPushConstants(cmd, m_raytracingPipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::PushConstant), &m_pushConst);

    const nvvk::SBTGenerator::Regions bindingTables = m_sbt.getSBTRegions();
    const VkExtent2D&                 size          = m_app->getViewportSize();
    vkCmdTraceRaysKHR(cmd, &bindingTables.raygen, &bindingTables.miss, &bindingTables.hit, &bindingTables.callable,
                      size.width * m_sizeMultiplier, size.height * m_sizeMultiplier, 1);
  }

  // Post-processing for AO filtering, toon shading etc
  {
    auto                            timerSection = m_profilerVK.cmdFrameSection(cmd, "Compositing");
    const VkExtent2D&               size         = m_app->getViewportSize();
    shaderio::CompositingShaderData compositingShaderData;
    compositingShaderData.windowSize = 3;
    compositingShaderData.frameIndex = m_animationShaderData.frameIndex;
    if(m_compositingPipeline != VK_NULL_HANDLE)
    {
      for(uint32_t i = 0; i < 4; i++)
      {
        compositingShaderData.passId = i;
        {
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_compositingPipeline);
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_compositingPipelineLayout, 0, 1,
                                  m_compositingDescriptorPack.sets.data(), 0, nullptr);
          vkCmdPushConstants(cmd, m_compositingPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                             sizeof(shaderio::CompositingShaderData), &compositingShaderData);

          uint32_t threadCount = size.width * size.height * m_sizeMultiplier * m_sizeMultiplier;

          uint32_t groupCount = uint32_t((threadCount + COMPOSITING_SHADER_BLOCK_SIZE - 1) / COMPOSITING_SHADER_BLOCK_SIZE);

          vkCmdDispatch(cmd, groupCount, 1, 1);

          memoryBarrier(cmd);
        }
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
    m_alloc.destroyBuffer(m.vertices);
    m_alloc.destroyBuffer(m.indices);
  }
  m_bMeshes.clear();

  m_alloc.destroyBuffer(m_bFrameInfo);
  m_alloc.destroyBuffer(m_bInstInfoBuffer);
  m_alloc.destroyBuffer(m_bMaterials);
  m_alloc.destroyBuffer(m_bSkyParams);


  if(!simpleReset)
  {
    m_gBuffers.deinit();
  }

  vkDestroyPipeline(m_device, m_raytracingPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_raytracingPipelineLayout, nullptr);
  m_raytracingPipeline = VK_NULL_HANDLE;
  m_sbt.deinit();


  vkDestroyPipeline(m_device, m_compositingPipeline, nullptr);
  m_compositingPipeline = VK_NULL_HANDLE;
  vkDestroyPipelineLayout(m_device, m_compositingPipelineLayout, nullptr);

  for(uint32_t i = 0; i < eAnimShaderCount; i++)
  {
    vkDestroyPipeline(m_device, m_animationPipelines[i], nullptr);
    m_animationPipelines[i] = VK_NULL_HANDLE;
  }
  vkDestroyPipelineLayout(m_device, m_animationPipelineLayout, nullptr);


  for(auto& b : m_blas)
  {
    m_alloc.destroyAcceleration(b);
  }
  m_blas.clear();
  m_alloc.destroyAcceleration(m_tlas);

  nvvk::PartitionedAccelerationStructure::Buffers ptlas = m_ptlas.getBuffers();

  m_alloc.destroyBuffer(ptlas.accelerationStructure);
  m_alloc.destroyBuffer(ptlas.buildScratch);
  m_alloc.destroyBuffer(ptlas.instanceUpdateInfo);
  m_alloc.destroyBuffer(ptlas.instanceWriteInfo);
  m_alloc.destroyBuffer(ptlas.operationsCount);
  m_alloc.destroyBuffer(ptlas.operationsInfo);
  m_alloc.destroyBuffer(ptlas.partitionWriteInfo);
  m_alloc.destroyBuffer(ptlas.updateScratch);

  m_alloc.destroyBuffer(m_globalState);
  m_alloc.destroyBuffer(m_globalStateHost);

  m_alloc.destroyBuffer(m_originalState);
  m_alloc.destroyBuffer(m_state[0]);
  m_alloc.destroyBuffer(m_state[1]);
  m_alloc.destroyBuffer(m_partitionState);

  m_alloc.destroyBuffer(m_partitionedTlasInstanceWriteDynamic);

  m_alloc.destroyBuffer(m_instancesBuffer);

  m_alloc.destroyBuffer(m_tlasScratchBuffer);

  m_alloc.destroyBuffer(m_sbtBuffer);

  m_ptlas.deinit();

  m_raytracingDescriptorPack.deinit();
  m_compositingDescriptorPack.deinit();

  if(!simpleReset)
  {
    m_stagingUploader.deinit();
    m_alloc.deinit();
    m_profilerVK.deinit();
    m_samplerPool.deinit();
    g_profilerManager.destroyTimeline(m_profilerTimeline);
  }
}


//////////////////////////////////////////////////////////////////////////
///
//////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{

  nvvk::Context                vkContext;  // The Vulkan context
  nvvk::ContextInitInfo        vkSetup;    // Information to create the Vulkan context
  nvapp::ApplicationCreateInfo appInfo;

  // Default application setting
  appInfo.vSync = false;

  // Parsing command line parameters
  nvutils::ParameterRegistry parameterRegistry;


  vkSetup.apiVersion = VK_API_VERSION_1_4;
  // clang-format off
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR };
    VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR computeDerivativesFeature{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR };
    VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR baryFeatures{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR };
    VkPhysicalDeviceNestedCommandBufferFeaturesEXT nestedCmdFeature{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_NESTED_COMMAND_BUFFER_FEATURES_EXT };
    VkPhysicalDeviceRayQueryFeaturesKHR rayqueryFeature{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR };
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR };
    VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeatures{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT };
    VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV reorderFeature{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV };
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFeatures{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT };
    VkPhysicalDeviceShaderClockFeaturesKHR clockFeatures{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR };
    VkPhysicalDevicePartitionedAccelerationStructureFeaturesNV ptlasFeature{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PARTITIONED_ACCELERATION_STRUCTURE_FEATURES_NV };
    VkPhysicalDeviceOpacityMicromapFeaturesEXT mm_opacity_features{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_FEATURES_EXT };
  // clang-format on

  // Requesting the extensions and features needed
  vkSetup.instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
  vkSetup.deviceExtensions   = {
      {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME},
      {VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME},
      {VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME},
      {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME},
      {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accelFeature},
      {VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rtPipelineFeature},
      {VK_KHR_RAY_QUERY_EXTENSION_NAME, &rayqueryFeature},
      {VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME, &computeDerivativesFeature},
      {VK_EXT_SHADER_OBJECT_EXTENSION_NAME, &shaderObjectFeatures},
      {VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME, &baryFeatures},
      {VK_EXT_NESTED_COMMAND_BUFFER_EXTENSION_NAME, &nestedCmdFeature},
      {VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME, &reorderFeature, false},  // Shading Execution Reorder
      {VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, &atomicFeatures},                   // Atomic float support
      {VK_KHR_SHADER_CLOCK_EXTENSION_NAME, &clockFeatures},                           // Shader timing
      {VK_NV_PARTITIONED_ACCELERATION_STRUCTURE_EXTENSION_NAME, &ptlasFeature,        // Partition TLAS
         VK_NV_PARTITIONED_ACCELERATION_STRUCTURE_SPEC_VERSION, true},                  //
  };
  if(!appInfo.headless)
  {
    nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }


#if defined(USE_NSIGHT_AFTERMATH)
  // Adding the Aftermath extension to the device and initialize the Aftermath
  auto& aftermath = AftermathCrashTracker::getInstance();
  aftermath.initialize();
  aftermath.addExtensions(vkSetup.deviceExtensions);
  // The callback function is called when a validation error is triggered. This will wait to give time to dump the GPU crash.
  nvvk::CheckError::getInstance().setCallbackFunction([&](VkResult result) { aftermath.errorCallback(result); });
#endif

  vkSetup.enableValidationLayers = true;
  nvvk::ValidationSettings validation{};
  validation.setPreset(nvvk::ValidationSettings::LayerPresets::eStandard);
  validation.printf_to_stdout  = VK_TRUE;
  validation.message_id_filter = {"VUID-VkWriteDescriptorSet-descriptorType-00319",
                                  "Undefined-Value-StorageImage-FormatMismatch-ImageView"};
  validation.unique_handles = VK_FALSE;  // Disable unique handles, as we use the same descriptor set for multiple pipelines
  vkSetup.instanceCreateInfoExt = validation.buildPNextChain();

  // Create the Vulkan context
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Error in Vulkan context creation\n");
    return 1;
  }


  // Application creation setup
  appInfo.name           = fmt::format("{}", PROJECT_NAME);
  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();


  // Create application
  nvapp::Application application;
  application.init(appInfo);


  auto partitionedTlas = std::make_shared<PartitionedTlasSample>();


  // #PROFILER
  // setup the profiler element and views
  nvapp::ElementProfiler::ViewSettings viewSettings{.name       = "Profiler",
                                                    .defaultTab = nvapp::ElementProfiler::TABLE,
                                                    .pieChart   = {.cpuTotal = false, .levels = 2},
                                                    .lineChart  = {.cpuLine = false}};
  // setting are optional, but can be used to expose to sample code (like hiding views for benchmark)

  auto elementProfiler = std::make_shared<nvapp::ElementProfiler>(
      &g_profilerManager, std::make_shared<nvapp::ElementProfiler::ViewSettings>(std::move(viewSettings)));


  // Add all application elements
  application.addElement(std::make_shared<nvapp::ElementDefaultMenu>());  // Menu / Quit


  g_cameraManipulator = std::make_shared<nvutils::CameraManipulator>();


  auto elemCamera = std::make_shared<nvapp::ElementCamera>();
  elemCamera->setCameraManipulator(g_cameraManipulator);
  application.addElement(elemCamera);

  application.addElement(std::make_shared<nvapp::ElementDefaultWindowTitle>());  // Window title info
  application.addElement(elementProfiler);
#ifdef USE_NVVK_INSPECTOR
  g_elementInspector = std::make_shared<nvapp::ElementInspector>();
  application.addElement(g_elementInspector);
#endif

  application.addElement(partitionedTlas);

  application.setVsync(true);
  application.run();

  application.deinit();
  vkContext.deinit();

  return 0;
}
