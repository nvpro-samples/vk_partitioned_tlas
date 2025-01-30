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

void PartitionedTlasSample::recompilePipeline()
{
  LOGI("Compiling raytracing pipeline\n");

  VkPipeline result = VK_NULL_HANDLE;

  // Creating all shaders
  for(auto& s : m_stages)
    s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;


  nvvk::ShaderModuleID rgenId = m_shaderManager.createShaderModule(VK_SHADER_STAGE_RAYGEN_BIT_KHR, "raytrace.rgen.glsl");
  m_stages[eRaygen].module = m_shaderManager.getShaderModule(rgenId).module;
  m_stages[eRaygen].pName  = "main";
  m_stages[eRaygen].stage  = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

  nvvk::ShaderModuleID missId = m_shaderManager.createShaderModule(VK_SHADER_STAGE_MISS_BIT_KHR, "raytrace.rmiss.glsl");
  m_stages[eMiss].module = m_shaderManager.getShaderModule(missId).module;
  m_stages[eMiss].pName  = "main";
  m_stages[eMiss].stage  = VK_SHADER_STAGE_MISS_BIT_KHR;

  nvvk::ShaderModuleID chitId = m_shaderManager.createShaderModule(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, "raytrace.rchit.glsl");
  m_stages[eClosestHit].module = m_shaderManager.getShaderModule(chitId).module;
  m_stages[eClosestHit].pName  = "main";
  m_stages[eClosestHit].stage  = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

  if(m_stages[eRaygen].module != VK_NULL_HANDLE)
  {
    m_dutil->setObjectName(m_stages[eRaygen].module, "Raygen");
  }
  if(m_stages[eMiss].module != VK_NULL_HANDLE)
  {
    m_dutil->setObjectName(m_stages[eMiss].module, "Miss");
  }
  if(m_stages[eClosestHit].module != VK_NULL_HANDLE)
  {
    m_dutil->setObjectName(m_stages[eClosestHit].module, "Closest Hit");
  }


  // Shader groups
  VkRayTracingShaderGroupCreateInfoKHR group{.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                                             .generalShader      = VK_SHADER_UNUSED_KHR,
                                             .closestHitShader   = VK_SHADER_UNUSED_KHR,
                                             .anyHitShader       = VK_SHADER_UNUSED_KHR,
                                             .intersectionShader = VK_SHADER_UNUSED_KHR};


  // Raygen
  m_shaderGroups[eRaygen]               = group;
  m_shaderGroups[eRaygen].type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  m_shaderGroups[eRaygen].generalShader = eRaygen;

  // Miss
  m_shaderGroups[eMiss]               = group;
  m_shaderGroups[eMiss].type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  m_shaderGroups[eMiss].generalShader = eMiss;

  // closest hit shader
  m_shaderGroups[eClosestHit]                  = group;
  m_shaderGroups[eClosestHit].type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
  m_shaderGroups[eClosestHit].generalShader    = VK_SHADER_UNUSED_KHR;
  m_shaderGroups[eClosestHit].closestHitShader = eClosestHit;

  m_rayPipelineInfo = {
      .sType                        = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
      .stageCount                   = static_cast<uint32_t>(m_stages.size()),  // Stages are shader
      .pStages                      = m_stages.data(),
      .groupCount                   = static_cast<uint32_t>(m_shaderGroups.size()),
      .pGroups                      = m_shaderGroups.data(),
      .maxPipelineRayRecursionDepth = MAXRAYRECURSIONDEPTH,  // Ray dept
      .layout                       = m_rtPipe.layout,
  };

  bool compilationSucceeded = true;
  for(auto& s : m_stages)
  {
    if(s.module == VK_NULL_HANDLE)
    {
      compilationSucceeded = false;
      break;
    }
  }
  if(compilationSucceeded)
  {
    vkCreateRayTracingPipelinesKHR(m_device, {}, {}, 1, &m_rayPipelineInfo, nullptr, &result);

    if(result != VK_NULL_HANDLE)
    {
      vkDeviceWaitIdle(m_device);
      vkDestroyPipeline(m_device, m_rtPipe.plines[0], nullptr);
      m_rtPipe.plines[0] = result;
      m_dutil->DBG_NAME(m_rtPipe.plines[0]);

      m_sbt.create(m_rtPipe.plines[0], m_rayPipelineInfo);
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

  nvvkhl::PipelineContainer& p = m_rtPipe;
  p.plines.resize(1);

  // This descriptor set, holds the top level acceleration structure and the output image
  // Create Binding Set
  m_rtSet->addBinding(B_tlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
  m_rtSet->addBinding(B_outImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
  m_rtSet->addBinding(B_aoImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
  m_rtSet->addBinding(B_depthImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
  m_rtSet->addBinding(B_frameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
  m_rtSet->addBinding(B_sceneDesc, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  m_rtSet->addBinding(B_skyParam, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
  m_rtSet->addBinding(B_materials, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  m_rtSet->addBinding(B_instances, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  m_rtSet->addBinding(B_vertex, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)m_bMeshes.size(), VK_SHADER_STAGE_ALL);
  m_rtSet->addBinding(B_index, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)m_bMeshes.size(), VK_SHADER_STAGE_ALL);
  m_rtSet->initLayout();
  m_rtSet->initPool(1);

  m_dutil->DBG_NAME(m_rtSet->getLayout());
  m_dutil->DBG_NAME(m_rtSet->getSet(0));

  // Push constant: we want to be able to update constants used by the shaders
  const VkPushConstantRange push_constant{VK_SHADER_STAGE_ALL, 0, sizeof(DH::PushConstant)};

  // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
  std::vector<VkDescriptorSetLayout> rt_desc_set_layouts = {m_rtSet->getLayout()};
  VkPipelineLayoutCreateInfo         pipeline_layout_create_info{
              .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
              .setLayoutCount         = static_cast<uint32_t>(rt_desc_set_layouts.size()),
              .pSetLayouts            = rt_desc_set_layouts.data(),
              .pushConstantRangeCount = 1,
              .pPushConstantRanges    = &push_constant,
  };
  vkCreatePipelineLayout(m_device, &pipeline_layout_create_info, nullptr, &p.layout);
  m_dutil->DBG_NAME(p.layout);

  recompilePipeline();
}

void PartitionedTlasSample::writeRtDesc()
{
  // Write to descriptors
  VkAccelerationStructureKHR tlas = m_tlas.accel;

  VkWriteDescriptorSetAccelerationStructureKHR desc_as_info{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
                                                            .accelerationStructureCount = 1,
                                                            .pAccelerationStructures    = &tlas};


  const VkDescriptorImageInfo  imageInfo{{}, m_gBuffers->getColorImageView(0), VK_IMAGE_LAYOUT_GENERAL};
  const VkDescriptorImageInfo  aoInfo{{}, m_gBuffers->getColorImageView(1), VK_IMAGE_LAYOUT_GENERAL};
  const VkDescriptorImageInfo  depthInfo{{}, m_gBuffers->getColorImageView(2), VK_IMAGE_LAYOUT_GENERAL};
  const VkDescriptorBufferInfo dbi_unif{m_bFrameInfo.buffer, 0, VK_WHOLE_SIZE};
  const VkDescriptorBufferInfo dbi_sky{m_bSkyParams.buffer, 0, VK_WHOLE_SIZE};
  const VkDescriptorBufferInfo mat_desc{m_bMaterials.buffer, 0, VK_WHOLE_SIZE};
  const VkDescriptorBufferInfo inst_desc{m_bInstInfoBuffer.buffer, 0, VK_WHOLE_SIZE};

  std::vector<VkDescriptorBufferInfo> vertex_desc;
  std::vector<VkDescriptorBufferInfo> index_desc;
  vertex_desc.reserve(m_bMeshes.size());
  index_desc.reserve(m_bMeshes.size());
  for(auto& m : m_bMeshes)
  {
    vertex_desc.push_back({m.vertices.buffer, 0, VK_WHOLE_SIZE});
    index_desc.push_back({m.indices.buffer, 0, VK_WHOLE_SIZE});
  }

  std::vector<VkWriteDescriptorSet> writes;


  if(m_animationShaderData.ptlasActive == 0)
  {
    writes.emplace_back(m_rtSet->makeWrite(0, B_tlas, &desc_as_info));
  }
  else
  {
    VkWriteDescriptorSet ptlasWrite = m_ptlas.getWriteDescriptorSet(m_rtSet->getSet(), B_tlas);
    writes.emplace_back(ptlasWrite);
  }

  writes.emplace_back(m_rtSet->makeWrite(0, B_outImage, &imageInfo));
  writes.emplace_back(m_rtSet->makeWrite(0, B_frameInfo, &dbi_unif));
  writes.emplace_back(m_rtSet->makeWrite(0, B_skyParam, &dbi_sky));
  writes.emplace_back(m_rtSet->makeWrite(0, B_materials, &mat_desc));
  writes.emplace_back(m_rtSet->makeWrite(0, B_instances, &inst_desc));
  writes.emplace_back(m_rtSet->makeWriteArray(0, B_vertex, vertex_desc.data()));
  writes.emplace_back(m_rtSet->makeWriteArray(0, B_index, index_desc.data()));
  writes.emplace_back(m_rtSet->makeWrite(0, B_aoImage, &aoInfo));
  writes.emplace_back(m_rtSet->makeWrite(0, B_depthImage, &depthInfo));

  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}


void PartitionedTlasSample::recompileAuxShaders()
{
  {
    LOGI("Compiling animation pipelines\n");

    std::array<nvvk::ShaderModuleID, eAnimShaderCount> animShaderIds;

    animShaderIds[eAnimInit] = m_shaderManager.createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "animation_init.comp.glsl");
    animShaderIds[eAnimPhysics] = m_shaderManager.createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "animation_physics.comp.glsl");
    animShaderIds[eAnimUpdateInstances] =
        m_shaderManager.createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "animation_update_instances.comp.glsl");

    m_animDispatcher.deinit();
    m_animDispatcher.init(m_device);
    for(size_t i = 0; i < animShaderIds.size(); i++)
    {
      if(animShaderIds[i].isValid())
      {
        VkShaderModule shaderModule = m_shaderManager.getShaderModule(animShaderIds[i]).module;
        if(shaderModule != VK_NULL_HANDLE)
        {
          m_animDispatcher.setCode(shaderModule, uint32_t(i));
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
    m_animDispatcher.finalizePipeline();
  }


  {
    LOGI("Compiling compositing pipeline\n");
    nvvk::ShaderModuleID compId = m_shaderManager.createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "compositing.comp.glsl");
    if(compId.isValid())
    {
      VkShaderModule shaderModule = m_shaderManager.getShaderModule(compId).module;
      if(shaderModule != VK_NULL_HANDLE)
      {
        m_compositingDispatcher.deinit();
        m_compositingDispatcher.init(m_device);
        m_compositingDispatcher.getBindings().addBinding(B_outImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
        m_compositingDispatcher.getBindings().addBinding(B_aoImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
        m_compositingDispatcher.getBindings().addBinding(B_depthImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
        m_compositingDispatcher.setCode(shaderModule);
        m_compositingDispatcher.finalizePipeline();
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
