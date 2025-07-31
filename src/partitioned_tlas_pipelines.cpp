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

#include "partitioned_tlas.hpp"
#include "nvutils/spirv.hpp"

void PartitionedTlasSample::recompilePipeline()
{
  LOGI("Compiling raytracing pipeline\n");

  VkPipeline result = VK_NULL_HANDLE;

  // Creating all shaders


  std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};

  std::array<VkRayTracingShaderGroupCreateInfoKHR, eShaderGroupCount> shaderGroups{};

  for(VkPipelineShaderStageCreateInfo& s : stages)
  {
    s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  }

  std::string rgenName  = "raytrace.rgen.glsl";
  std::string rchitName = "raytrace.rchit.glsl";
  std::string rmissName = "raytrace.rmiss.glsl";

  LOGI("\t%s\n", rgenName.c_str());


  shaderc::SpvCompilationResult rgenId = m_glslCompiler.compileFile(rgenName, shaderc_glsl_raygen_shader);
  assert(m_glslCompiler.isValid(rgenId));
  nvutils::dumpSpirvWithHashedName(rgenName + ".spv", m_glslCompiler.getSpirvData(rgenId).data(),
                                   m_glslCompiler.getSpirvSize(rgenId));


  VkShaderModuleCreateInfo rgenShaderInfo = m_glslCompiler.makeShaderModuleCreateInfo(rgenId);
  stages[eRaygen].pNext                   = &rgenShaderInfo;
  stages[eRaygen].pName                   = "main";
  stages[eRaygen].stage                   = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

  LOGI("\t%s\n", rmissName.c_str());
  shaderc::SpvCompilationResult missId = m_glslCompiler.compileFile(rmissName, shaderc_glsl_miss_shader);
  assert(m_glslCompiler.isValid(missId));
  nvutils::dumpSpirvWithHashedName(rmissName + ".spv", m_glslCompiler.getSpirvData(missId).data(),
                                   m_glslCompiler.getSpirvSize(missId));


  VkShaderModuleCreateInfo rmissShaderInfo = m_glslCompiler.makeShaderModuleCreateInfo(missId);
  stages[eMiss].pNext                      = &rmissShaderInfo;
  stages[eMiss].pName                      = "main";
  stages[eMiss].stage                      = VK_SHADER_STAGE_MISS_BIT_KHR;

  LOGI("\t%s\n", rchitName.c_str());
  shaderc::SpvCompilationResult chitId = m_glslCompiler.compileFile(rchitName, shaderc_glsl_closesthit_shader);
  assert(m_glslCompiler.isValid(chitId));
  nvutils::dumpSpirvWithHashedName(rchitName + ".spv", m_glslCompiler.getSpirvData(chitId).data(),
                                   m_glslCompiler.getSpirvSize(chitId));
  VkShaderModuleCreateInfo rchitShaderInfo = m_glslCompiler.makeShaderModuleCreateInfo(chitId);
  stages[eClosestHit].pNext                = &rchitShaderInfo;
  stages[eClosestHit].pName                = "main";
  stages[eClosestHit].stage                = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

  // Shader groups
  VkRayTracingShaderGroupCreateInfoKHR group{.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                                             .generalShader      = VK_SHADER_UNUSED_KHR,
                                             .closestHitShader   = VK_SHADER_UNUSED_KHR,
                                             .anyHitShader       = VK_SHADER_UNUSED_KHR,
                                             .intersectionShader = VK_SHADER_UNUSED_KHR};


  // Raygen
  shaderGroups[eRaygen]               = group;
  shaderGroups[eRaygen].type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  shaderGroups[eRaygen].generalShader = eRaygen;

  // Miss
  shaderGroups[eMiss]               = group;
  shaderGroups[eMiss].type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  shaderGroups[eMiss].generalShader = eMiss;

  // closest hit shader
  shaderGroups[eClosestHit]                  = group;
  shaderGroups[eClosestHit].type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
  shaderGroups[eClosestHit].generalShader    = VK_SHADER_UNUSED_KHR;
  shaderGroups[eClosestHit].closestHitShader = eClosestHit;

  m_raytracingPipelineInfo = {
      .sType                        = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
      .flags                        = {},
      .stageCount                   = uint32_t(stages.size()),  // Stages are shader
      .pStages                      = stages.data(),
      .groupCount                   = uint32_t(shaderGroups.size()),
      .pGroups                      = shaderGroups.data(),
      .maxPipelineRayRecursionDepth = MAXRAYRECURSIONDEPTH,  // Ray dept
      .layout                       = m_raytracingPipelineLayout,
  };


  bool compilationSucceeded = true;
  if(compilationSucceeded)
  {
    vkCreateRayTracingPipelinesKHR(m_device, {}, {}, 1, &m_raytracingPipelineInfo, nullptr, &result);

    if(result != VK_NULL_HANDLE)
    {
      vkDeviceWaitIdle(m_device);
      vkDestroyPipeline(m_device, m_raytracingPipeline, nullptr);
      m_raytracingPipeline = result;
      NVVK_DBG_NAME(m_raytracingPipeline);


      // Creating the SBT
      {
        // Shader Binding Table (SBT) setup
        // Prepare SBT data from ray pipeline
        size_t bufferSize = m_sbt.calculateSBTBufferSize(m_raytracingPipeline, m_raytracingPipelineInfo);

        // Create SBT buffer using the size from above
        NVVK_CHECK(m_alloc.createBuffer(m_sbtBuffer, bufferSize, VK_BUFFER_USAGE_2_SHADER_BINDING_TABLE_BIT_KHR, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                        VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
                                        m_sbt.getBufferAlignment()));
        NVVK_DBG_NAME(m_sbtBuffer.buffer);


        // Pass the manual mapped pointer to fill the sbt data
        NVVK_CHECK(m_sbt.populateSBTBuffer(m_sbtBuffer.address, bufferSize, m_sbtBuffer.mapping));
      }

      return;
    }
  }
  m_invalidShaderNotified = false;
}

//--------------------------------------------------------------------------------------------------
// Pipeline for the ray tracer: all shaders, raygen, chit, miss
//
void PartitionedTlasSample::createRtxPipeline()
{
  m_raytracingDescriptorPack.deinit();
  m_raytracingDescriptorPack.bindings.addBinding(B_outImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
  m_raytracingDescriptorPack.bindings.addBinding(B_aoImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
  m_raytracingDescriptorPack.bindings.addBinding(B_depthImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);


  m_raytracingDescriptorPack.bindings.addBinding(B_frameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
  m_raytracingDescriptorPack.bindings.addBinding(B_sceneDesc, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  m_raytracingDescriptorPack.bindings.addBinding(B_skyParam, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);

  m_raytracingDescriptorPack.bindings.addBinding(B_materials, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  m_raytracingDescriptorPack.bindings.addBinding(B_instances, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  m_raytracingDescriptorPack.bindings.addBinding(B_vertex, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                 (uint32_t)m_bMeshes.size(), VK_SHADER_STAGE_ALL);
  m_raytracingDescriptorPack.bindings.addBinding(B_index, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)m_bMeshes.size(),
                                                 VK_SHADER_STAGE_ALL);


  m_raytracingDescriptorPack.initFromBindings(m_device);
  NVVK_DBG_NAME(m_raytracingDescriptorPack.layout);
  NVVK_DBG_NAME(m_raytracingDescriptorPack.sets[0]);

  // Push constant: we want to be able to update constants used by the shaders
  const VkPushConstantRange pushConstant{VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::PushConstant)};

  // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
  std::vector<VkDescriptorSetLayout> raytracingDescriptorLayouts = {m_raytracingDescriptorPack.layout};
  VkPipelineLayoutCreateInfo         pipelineLayoutCreateInfo{
              .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
              .setLayoutCount         = uint32_t(raytracingDescriptorLayouts.size()),
              .pSetLayouts            = raytracingDescriptorLayouts.data(),
              .pushConstantRangeCount = 1,
              .pPushConstantRanges    = &pushConstant,
  };
  vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr, &m_raytracingPipelineLayout);
  NVVK_DBG_NAME(m_raytracingPipelineLayout);

  recompilePipeline();
}

void PartitionedTlasSample::writeRaytracingDescriptors()
{
  // Write to descriptors
  VkAccelerationStructureKHR tlas = m_tlas.accel;


  nvvk::WriteSetContainer writes;


  const VkDescriptorImageInfo  imageInfo{{}, m_gBuffers.getColorImageView(0), VK_IMAGE_LAYOUT_GENERAL};
  const VkDescriptorImageInfo  aoInfo{{}, m_gBuffers.getColorImageView(1), VK_IMAGE_LAYOUT_GENERAL};
  const VkDescriptorImageInfo  depthInfo{{}, m_gBuffers.getColorImageView(2), VK_IMAGE_LAYOUT_GENERAL};
  const VkDescriptorBufferInfo frameInfo{m_bFrameInfo.buffer, 0, VK_WHOLE_SIZE};
  const VkDescriptorBufferInfo skyInfo{m_bSkyParams.buffer, 0, VK_WHOLE_SIZE};
  const VkDescriptorBufferInfo matInfo{m_bMaterials.buffer, 0, VK_WHOLE_SIZE};
  const VkDescriptorBufferInfo instanceInfo{m_bInstInfoBuffer.buffer, 0, VK_WHOLE_SIZE};

  std::vector<VkDescriptorBufferInfo> vertexInfos;
  std::vector<VkDescriptorBufferInfo> indexInfos;
  vertexInfos.reserve(m_bMeshes.size());
  vertexInfos.reserve(m_bMeshes.size());
  indexInfos.reserve(m_bMeshes.size());
  for(PrimitiveMeshVk& m : m_bMeshes)
  {
    vertexInfos.push_back({m.vertices.buffer, 0, VK_WHOLE_SIZE});
    indexInfos.push_back({m.indices.buffer, 0, VK_WHOLE_SIZE});
  }


  writes.append(m_raytracingDescriptorPack.bindings.getWriteSet(B_outImage, m_raytracingDescriptorPack.sets[0]), &imageInfo);
  writes.append(m_raytracingDescriptorPack.bindings.getWriteSet(B_frameInfo, m_raytracingDescriptorPack.sets[0]), &frameInfo);
  writes.append(m_raytracingDescriptorPack.bindings.getWriteSet(B_skyParam, m_raytracingDescriptorPack.sets[0]), &skyInfo);
  writes.append(m_raytracingDescriptorPack.bindings.getWriteSet(B_materials, m_raytracingDescriptorPack.sets[0]), &matInfo);


  writes.append(m_raytracingDescriptorPack.bindings.getWriteSet(B_instances, m_raytracingDescriptorPack.sets[0]), &instanceInfo);

  for(size_t i = 0; i < m_bMeshes.size(); i++)
  {
    writes.append(m_raytracingDescriptorPack.bindings.getWriteSet(B_vertex, m_raytracingDescriptorPack.sets[0], uint32_t(i)),
                  &vertexInfos[i]);
    writes.append(m_raytracingDescriptorPack.bindings.getWriteSet(B_index, m_raytracingDescriptorPack.sets[0], uint32_t(i)),
                  &indexInfos[i]);
  }

  writes.append(m_raytracingDescriptorPack.bindings.getWriteSet(B_aoImage, m_raytracingDescriptorPack.sets[0]), &aoInfo);
  writes.append(m_raytracingDescriptorPack.bindings.getWriteSet(B_depthImage, m_raytracingDescriptorPack.sets[0]), &depthInfo);

  vkUpdateDescriptorSets(m_device, uint32_t(writes.size()), writes.data(), 0, nullptr);
}

void PartitionedTlasSample::writeCompositingDescriptors()
{
  nvvk::WriteSetContainer writes;

  const VkDescriptorImageInfo imageInfo{{}, m_gBuffers.getColorImageView(0), VK_IMAGE_LAYOUT_GENERAL};
  writes.append(m_compositingDescriptorPack.bindings.getWriteSet(B_outImage, m_compositingDescriptorPack.sets[0]), &imageInfo);

  const VkDescriptorImageInfo aoImageInfo{{}, m_gBuffers.getColorImageView(1), VK_IMAGE_LAYOUT_GENERAL};
  writes.append(m_compositingDescriptorPack.bindings.getWriteSet(B_aoImage, m_compositingDescriptorPack.sets[0]), &aoImageInfo);

  const VkDescriptorImageInfo depthImageInfo{{}, m_gBuffers.getColorImageView(2), VK_IMAGE_LAYOUT_GENERAL};
  writes.append(m_compositingDescriptorPack.bindings.getWriteSet(B_depthImage, m_compositingDescriptorPack.sets[0]), &depthImageInfo);

  vkUpdateDescriptorSets(m_device, uint32_t(writes.size()), writes.data(), 0, nullptr);
}


void PartitionedTlasSample::recompileAuxShaders()
{
  {
    LOGI("Compiling animation pipelines\n");

    LOGI("\t%s\n", "shaders/animation_init.comp.glsl");
    m_animationShaderModules[eAnimInit] =
        m_glslCompiler.compileFile("shaders/animation_init.comp.glsl", shaderc_glsl_compute_shader);
    nvutils::dumpSpirvWithHashedName("spvdump/animation_init.comp.spv",
                                     m_glslCompiler.getSpirvData(m_animationShaderModules[eAnimInit]).data(),
                                     m_glslCompiler.getSpirvSize(m_animationShaderModules[eAnimInit]));

    LOGI("\t%s\n", "shaders/animation_physics.comp.glsl");
    m_animationShaderModules[eAnimPhysics] =
        m_glslCompiler.compileFile("shaders/animation_physics.comp.glsl", shaderc_glsl_compute_shader);
    nvutils::dumpSpirvWithHashedName("spvdump/animation_physics.comp.spv",
                                     m_glslCompiler.getSpirvData(m_animationShaderModules[eAnimPhysics]).data(),
                                     m_glslCompiler.getSpirvSize(m_animationShaderModules[eAnimPhysics]));
    LOGI("\t%s\n", "shaders/animation_instances.comp.glsl");
    m_animationShaderModules[eAnimUpdateInstances] =
        m_glslCompiler.compileFile("shaders/animation_update_instances.comp.glsl", shaderc_glsl_compute_shader);
    nvutils::dumpSpirvWithHashedName("spvdump/animation_update_instances.comp.spv",
                                     m_glslCompiler.getSpirvData(m_animationShaderModules[eAnimUpdateInstances]).data(),
                                     m_glslCompiler.getSpirvSize(m_animationShaderModules[eAnimUpdateInstances]));

    for(uint32_t i = 0; i < eAnimShaderCount; i++)
    {
      if(m_animationShaderModules[i].GetNumErrors() > 0)
      {
        m_invalidShaderNotified = false;
      }
    }


    {
      VkPushConstantRange pushRange;
      pushRange.offset     = 0;
      pushRange.size       = sizeof(shaderio::AnimationShaderData);
      pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

      VkPipelineLayoutCreateInfo layoutCreateInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
      layoutCreateInfo.setLayoutCount             = 0;
      layoutCreateInfo.pSetLayouts                = nullptr;  //&m_compositingDescriptorPack.layout;
      layoutCreateInfo.pushConstantRangeCount     = 1;
      layoutCreateInfo.pPushConstantRanges        = &pushRange;
      NVVK_CHECK(vkCreatePipelineLayout(m_device, &layoutCreateInfo, nullptr, &m_animationPipelineLayout));
      NVVK_DBG_NAME(m_animationPipelineLayout);

      VkComputePipelineCreateInfo compInfo = {.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                              .stage  = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                         .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                                                         .pName = "main"},
                                              .layout = m_animationPipelineLayout};

      for(uint32_t i = 0; i < eAnimShaderCount; i++)
      {
        VkShaderModuleCreateInfo shaderInfo = m_glslCompiler.makeShaderModuleCreateInfo(m_animationShaderModules[i]);
        compInfo.stage.pNext                = &shaderInfo;
        NVVK_CHECK(vkCreateComputePipelines(m_device, nullptr, 1, &compInfo, nullptr, &m_animationPipelines[i]));
      }
    }
  }


  {
    LOGI("Compiling compositing pipeline\n");
    m_glslCompiler.defaultOptions();
    shaderc::SpvCompilationResult compId = m_glslCompiler.compileFile("shaders/compositing.comp.glsl", shaderc_glsl_compute_shader);


    if(compId.GetNumErrors() == 0)
    {
      VkShaderModuleCreateInfo shaderInfo = m_glslCompiler.makeShaderModuleCreateInfo(compId);

      VkShaderModule shaderModule{};
      NVVK_CHECK(vkCreateShaderModule(m_device, &shaderInfo, nullptr, &shaderModule));

      if(shaderModule != VK_NULL_HANDLE)
      {
        m_compositingDescriptorPack.deinit();
        m_compositingDescriptorPack.bindings.addBinding(B_outImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
        m_compositingDescriptorPack.bindings.addBinding(B_aoImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
        m_compositingDescriptorPack.bindings.addBinding(B_depthImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
        m_compositingDescriptorPack.initFromBindings(m_device);


        {
          VkPushConstantRange pushRange;
          pushRange.offset     = 0;
          pushRange.size       = sizeof(shaderio::CompositingShaderData);
          pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

          VkPipelineLayoutCreateInfo layoutCreateInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
          layoutCreateInfo.setLayoutCount             = 1;
          layoutCreateInfo.pSetLayouts                = &m_compositingDescriptorPack.layout;
          layoutCreateInfo.pushConstantRangeCount     = 1;
          layoutCreateInfo.pPushConstantRanges        = &pushRange;
          NVVK_CHECK(vkCreatePipelineLayout(m_device, &layoutCreateInfo, nullptr, &m_compositingPipelineLayout));
          NVVK_DBG_NAME(m_compositingPipelineLayout);

          VkComputePipelineCreateInfo compInfo = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                                  .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                                                            .pName = "main"},
                                                  .layout = m_compositingPipelineLayout};

          compInfo.stage.pNext = &shaderInfo;
          NVVK_CHECK(vkCreateComputePipelines(m_device, nullptr, 1, &compInfo, nullptr, &m_compositingPipeline));
        }


        vkDestroyShaderModule(m_device, shaderModule, nullptr);
      }
      else
      {
        m_invalidShaderNotified = false;
      }
    }
    else
    {
      m_invalidShaderNotified = false;
    }
  }
}
