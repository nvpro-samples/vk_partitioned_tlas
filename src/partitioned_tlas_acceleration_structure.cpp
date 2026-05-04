/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <unordered_map>

#include "nvutils/parallel_work.hpp"
#include "partitioned_tlas.hpp"
#include "partitioned_acceleration_structures.hpp"

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

//--------------------------------------------------------------------------------------------------
// Converting a PrimitiveMesh as input for BLAS
//
static nvvk::AccelerationStructureGeometryInfo primitiveToGeometry(const nvutils::PrimitiveMesh& prim,
                                                                   VkDeviceAddress               vertexAddress,
                                                                   VkDeviceAddress               indexAddress)
{
  nvvk::AccelerationStructureGeometryInfo result;
  const uint32_t                          triangleCount = uint32_t(prim.triangles.size());

  // Describe buffer as array of VertexObj.
  VkAccelerationStructureGeometryTrianglesDataKHR triangles{
      .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
      .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,  // vec3 vertex position data
      .vertexData   = {.deviceAddress = vertexAddress},
      .vertexStride = sizeof(nvutils::PrimitiveVertex),
      .maxVertex    = uint32_t(prim.vertices.size()) - 1,
      .indexType    = VK_INDEX_TYPE_UINT32,
      .indexData    = {.deviceAddress = indexAddress},
  };

  // Identify the above data as containing opaque triangles.
  result.geometry = VkAccelerationStructureGeometryKHR{
      .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
      .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
      .geometry     = {.triangles = triangles},
      .flags        = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR | VK_GEOMETRY_OPAQUE_BIT_KHR,
  };

  result.rangeInfo = VkAccelerationStructureBuildRangeInfoKHR{
      .primitiveCount  = triangleCount,
      .primitiveOffset = 0,
      .firstVertex     = 0,
      .transformOffset = 0,
  };

  return result;
}

//--------------------------------------------------------------------------------------------------
// Create all bottom level acceleration structures (BLAS) - This code is common to TLAS and PTLAS as the BLAS'es are exactly the same
//
void PartitionedTlasSample::createBottomLevelAS()
{
  nvutils::ScopedTimer st(__FUNCTION__);

  size_t meshCount = m_meshes.size();

  // BLAS - Storing each primitive in a geometry
  std::vector<nvvk::AccelerationStructureBuildData> blasBuildData(meshCount);
  m_blas.resize(meshCount);  // All BLAS

  // Get the build information for all the BLAS
  VkDeviceSize maxScratchSize = 0;
  for(uint32_t meshIndex = 0; meshIndex < meshCount; meshIndex++)
  {
    const VkDeviceAddress vertexAddress = m_bMeshes[meshIndex].vertices.address;
    const VkDeviceAddress indexAddress  = m_bMeshes[meshIndex].indices.address;
    blasBuildData[meshIndex].asType     = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    blasBuildData[meshIndex].addGeometry(primitiveToGeometry(m_meshes[meshIndex], vertexAddress, indexAddress));
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo =
        blasBuildData[meshIndex].finalizeGeometry(m_device, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
    maxScratchSize = std::max(maxScratchSize, sizeInfo.buildScratchSize);
  }

  // Create the scratch buffer
  nvvk::Buffer scratchBuffer;
  // Force proper alignment by using VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
  NVVK_CHECK(m_alloc.createBuffer(scratchBuffer, maxScratchSize,
                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                      | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR));

  // Create the acceleration structures
  VkCommandBuffer cmd = m_app->createTempCmdBuffer();
  {
    for(uint32_t meshIndex = 0; meshIndex < meshCount; meshIndex++)
    {
      NVVK_CHECK(m_alloc.createAcceleration(m_blas[meshIndex], blasBuildData[meshIndex].makeCreateInfo()));
      blasBuildData[meshIndex].cmdBuildAccelerationStructure(cmd, m_blas[meshIndex].accel, scratchBuffer.address);
    }
  }
  m_app->submitAndWaitTempCmdBuffer(cmd);

  // Cleanup
  m_alloc.destroyBuffer(scratchBuffer);
}

//--------------------------------------------------------------------------------------------------
// Create a regular top level acceleration structures from the scene BLAS'es
//
void PartitionedTlasSample::createTopLevelAS()
{
  nvutils::ScopedTimer st(__FUNCTION__);

  std::vector<VkAccelerationStructureInstanceKHR> tlasInstances;
  tlasInstances.resize(m_nodes.size());
  uint32_t threadCount = std::min((uint32_t)m_nodes.size(), std::thread::hardware_concurrency());
  nvutils::parallel_batches(
      m_nodes.size(),
      [&](uint64_t i) {
        nvutils::Node& node = m_nodes[i];

        VkAccelerationStructureInstanceKHR instance{
            .transform           = nvvk::toTransformMatrixKHR(node.localMatrix()),  // Position of the instance
            .instanceCustomIndex = uint32_t(node.mesh),                             // gl_InstanceCustomIndexEX
            .mask                = 0xFF,                                            // All objects
            .instanceShaderBindingTableRecordOffset = 0,  // We will use the same hit group for all object
            .flags                                  = {},
            .accelerationStructureReference         = m_blas[node.mesh].address,
        };
        tlasInstances[i] = instance;
      },
      threadCount);

  VkCommandBuffer cmd = m_app->createTempCmdBuffer();

  // Create the instances buffer, add a barrier to ensure the data is copied before the TLAS build
  NVVK_CHECK(m_alloc.createBuffer(m_instancesBuffer, std::span(tlasInstances).size_bytes(),
                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                      | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR));
  NVVK_CHECK(m_stagingUploader.appendBuffer(m_instancesBuffer, 0, std::span(tlasInstances)));
  NVVK_DBG_NAME(m_instancesBuffer.buffer);
  m_stagingUploader.cmdUploadAppended(cmd);

  nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);

  m_tlasBuildData.asType = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

  nvvk::AccelerationStructureGeometryInfo geometryInfo =
      m_tlasBuildData.makeInstanceGeometry(tlasInstances.size(), m_instancesBuffer.address);
  m_tlasBuildData.addGeometry(geometryInfo);
  // Get the size of the TLAS
  VkAccelerationStructureBuildSizesInfoKHR sizeInfo =
      m_tlasBuildData.finalizeGeometry(m_device, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                                     | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR);

  if(sizeInfo.accelerationStructureSize == ~VkDeviceSize(0) || sizeInfo.buildScratchSize == ~VkDeviceSize(0))
  {
    LOGE("Error: failed to get the size of the TLAS - Regular TLAS unavailable\n");
    return;
  }

  m_stats.tlasAccelerationStructureSize = sizeInfo.accelerationStructureSize;
  m_stats.tlasBuildScratchSize          = sizeInfo.buildScratchSize;

  // Create the scratch buffer
  NVVK_CHECK(m_alloc.createBuffer(m_tlasScratchBuffer, sizeInfo.buildScratchSize,
                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                      | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR));

  // Create the TLAS
  NVVK_CHECK(m_alloc.createAcceleration(m_tlas, m_tlasBuildData.makeCreateInfo()));
  m_tlasBuildData.cmdBuildAccelerationStructure(cmd, m_tlas.accel, m_tlasScratchBuffer.address);
  m_app->submitAndWaitTempCmdBuffer(cmd);
}

// Create a partitions acceleration structure from the scene BLAS'es
void PartitionedTlasSample::createPartitionedTopLevelAS()
{
  nvutils::ScopedTimer st(__FUNCTION__);
  // The partitions are defined on a simple grid
  uint32_t partitionCount = m_partitionCountPerAxis * m_partitionCountPerAxis + 1;  // +1 is for the ground plane

  uint32_t maxInstancesPerPartition = 0;


  // Build the instance information for each domino, and add them to their respective partitions
  std::vector<VkPartitionedAccelerationStructureWriteInstanceDataNV> instances;
  std::unordered_map<uint32_t, uint32_t>                             partitionSizes;

  instances.reserve(m_nodes.size());

  for(size_t i = 0; i < m_nodes.size(); i++)
  {
    uint32_t partitionID = m_partitionIndexPerNode[i];
    auto     it          = partitionSizes.find(partitionID);
    if(it != partitionSizes.end())
    {
      it->second++;
    }
    else
    {
      partitionSizes[partitionID] = 1;
    }
    instances.push_back({});
  }

  uint32_t threadCount = std::min((uint32_t)m_nodes.size(), std::thread::hardware_concurrency());


  nvutils::parallel_batches(
      m_nodes.size(),
      [&](uint64_t i) {
        nvutils::Node&                                        node = m_nodes[i];
        VkPartitionedAccelerationStructureWriteInstanceDataNV instanceData{};
        instanceData.transform                           = nvvk::toTransformMatrixKHR(node.localMatrix());
        instanceData.instanceID                          = uint32_t(i);
        instanceData.instanceMask                        = 0xFF;
        instanceData.instanceContributionToHitGroupIndex = 0;
        instanceData.instanceIndex                       = uint32_t(i);
        instanceData.accelerationStructure               = m_blas[node.mesh].buffer.address;
        instanceData.partitionIndex                      = m_partitionIndexPerNode[i];
        instances[i]                                     = instanceData;
      },
      threadCount);


  for(auto count : partitionSizes)
  {
    maxInstancesPerPartition = std::max(maxInstancesPerPartition, count.second);
  }

  m_ptlas.init(m_app->getDevice());


  nvvk::PartitionedAccelerationStructure::BuildSizeQuery sizeQuery{};
  sizeQuery.instanceCount             = uint32_t(m_nodes.size());
  sizeQuery.partitionCount            = partitionCount;
  sizeQuery.maxInstancesPerPartition  = maxInstancesPerPartition;
  sizeQuery.allowPartitionTranslation = false;
  // This sample only updates instances and does not use per-partition translation vectors, hence we only need one operation
  sizeQuery.maxOperations                 = 1;
  sizeQuery.maxInstancesInGlobalPartition = m_dynamicObjectCount;


  // Fetch the sizes of the buffers for PTLAS build and storage, as well as auxiliary buffers for instance and partition storage
  nvvk::PartitionedAccelerationStructure::BuildSizeInfo sizeInfo = m_ptlas.getBuildSizes(sizeQuery);

  m_stats.ptlasBuildScratchSize          = sizeInfo.buildScratchSize;
  m_stats.ptlasAccelerationStructureSize = sizeInfo.accelerationStructureSize;

  // Allocate the buffers for PTLAS storage
  nvvk::PartitionedAccelerationStructure::Buffers buffers;

  // We need to be able to get the address of all buffers here. In addition,
  // when the inspector is active, we need to be able to transfer from them.
  const VkBufferUsageFlags baseFlags = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
#ifdef USE_NVVK_INSPECTOR
                                       | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
#endif
      ;

  NVVK_CHECK(m_alloc.createBuffer(buffers.accelerationStructure, sizeInfo.accelerationStructureSize,
                                  baseFlags | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR));

  NVVK_CHECK(m_alloc.createBuffer(buffers.buildScratch, sizeInfo.buildScratchSize, baseFlags | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));

  if(sizeInfo.updateScratchSize != 0)
  {
    NVVK_CHECK(m_alloc.createBuffer(buffers.updateScratch, sizeInfo.updateScratchSize, baseFlags | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
  }


  NVVK_CHECK(m_alloc.createBuffer(buffers.operationsInfo, sizeInfo.operationInfoSize,
                                  baseFlags | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                                      | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR));

  NVVK_CHECK(m_alloc.createBuffer(buffers.operationsCount, sizeInfo.operationCountSize,
                                  baseFlags | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                                      | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR));

  NVVK_CHECK(m_alloc.createBuffer(buffers.instanceWriteInfo, sizeInfo.instanceWriteInfoSize,
                                  baseFlags | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT));

  if(sizeInfo.instanceUpdateInfoSize > 0)
  {
    NVVK_CHECK(m_alloc.createBuffer(buffers.instanceUpdateInfo, sizeInfo.instanceUpdateInfoSize,
                                    baseFlags | VK_BUFFER_USAGE_TRANSFER_DST_BIT));
  }

  if(sizeInfo.partitionWriteInfoSize > 0)
  {
    NVVK_CHECK(m_alloc.createBuffer(buffers.partitionWriteInfo, sizeInfo.partitionWriteInfoSize,
                                    baseFlags | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT));
  }


  m_ptlas.setBuffers(buffers);

  m_animationShaderData.originalInstanceWriteInfo = m_ptlas.getBuffers().instanceWriteInfo.address;
  m_animationShaderData.globalPartitionIndex      = partitionCount - 1;

  // Upload the instance and partition data, and perform the first PTLAS build

  {
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();

    m_ptlas.uploadPtlasData(&m_stagingUploader, cmd, instances);

    memoryBarrier(cmd);

    m_ptlas.buildAccelerationStructure(cmd);


    m_app->submitAndWaitTempCmdBuffer(cmd);
  }


  // Allocate a buffer to store the dynamic instance updates during animation

  NVVK_CHECK(m_alloc.createBuffer(m_partitionedTlasInstanceWriteDynamic, sizeInfo.instanceWriteInfoSize,
                                  baseFlags | VK_BUFFER_USAGE_TRANSFER_DST_BIT));

#ifdef USE_NVVK_INSPECTOR
  if(m_ptlas.getBuffers().partitionWriteInfo.buffer != VK_NULL_HANDLE)
  {
    nvapp::ElementInspector::BufferInspectionInfo info{};
    info.entryCount =
        uint32_t(sizeInfo.partitionWriteInfoSize / sizeof(VkPartitionedAccelerationStructureWritePartitionTranslationDataNV));
    info.format = g_elementInspector->formatStruct(
        "struct VkPartitionedAccelerationStructureWritePartitionTranslationDataNV\
  {\
    uint32_t        partitionIndex;\
    float       partitionTranslation[3];\
  }");
    info.name         = "partition write original";
    info.sourceBuffer = m_ptlas.getBuffers().partitionWriteInfo.buffer;
    g_elementInspector->initBufferInspection(ePartitionWriteOriginal, info);
  }


  {
    nvapp::ElementInspector::BufferInspectionInfo info{};
    info.entryCount = uint32_t(sizeInfo.instanceWriteInfoSize / sizeof(VkPartitionedAccelerationStructureWriteInstanceDataNV));
    info.format = g_elementInspector->formatStruct(
        "struct PartitionedAccelerationStructureWriteInstanceDataNV\
    {\
      mat3x4      transform;\
      float       explicitAABB[6];\
      uint32_t    instanceID;\
      uint32_t    instanceMask;\
      uint32_t    instanceContributionToHitGroupIndex;\
      uint32_t    instanceFlags;\
      uint32_t    instanceIndex;\
      uint32_t    partitionIndex;\
      uint64_t    accelerationStructure;\
    }");
    info.name         = "instance write";
    info.sourceBuffer = m_partitionedTlasInstanceWriteDynamic.buffer;
    g_elementInspector->initBufferInspection(eInstanceWrite, info);
  }

  {
    nvapp::ElementInspector::BufferInspectionInfo info{};
    info.entryCount = uint32_t(sizeInfo.instanceWriteInfoSize / sizeof(VkPartitionedAccelerationStructureWriteInstanceDataNV));
    info.format = g_elementInspector->formatStruct(
        "struct PartitionedAccelerationStructureWriteInstanceDataNV\
    {\
      mat3x4                                               transform;\
      float                                                explicitAABB[6];\
      uint32_t                                             instanceID;\
      uint32_t                                             instanceMask;\
      uint32_t                                             instanceContributionToHitGroupIndex;\
      uint32_t    instanceFlags;\
      uint32_t                                             instanceIndex;\
      uint32_t                                             partitionIndex;\
      uint64_t                                              accelerationStructure;\
    }");
    info.name         = "instance write original";
    info.sourceBuffer = m_ptlas.getBuffers().instanceWriteInfo.buffer;
    g_elementInspector->initBufferInspection(eInstanceWriteOriginal, info);
  }

  {

    nvapp::ElementInspector::BufferInspectionInfo info{};
    info.entryCount = uint32_t(sizeInfo.operationInfoSize / sizeof(VkBuildPartitionedAccelerationStructureIndirectCommandNV));
    info.format = g_elementInspector->formatStruct(
        "struct VkBuildPartitionedAccelerationStructureIndirectCommandNV\
    {\
      uint32_t opType;\
      uint32_t                                    argCount;\
      uint64_t                   argData;\
    }");
    info.name         = "ops";
    info.sourceBuffer = buffers.operationsInfo.buffer;
    g_elementInspector->initBufferInspection(eOps, info);
  }

#endif
}
