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

#include "nvh/parallel_work.hpp"
#include <unordered_map>

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
static nvvk::AccelerationStructureGeometryInfo primitiveToGeometry(const nvh::PrimitiveMesh& prim,
                                                                   VkDeviceAddress           vertexAddress,
                                                                   VkDeviceAddress           indexAddress)
{
  nvvk::AccelerationStructureGeometryInfo result;
  const auto                              triangleCount = static_cast<uint32_t>(prim.triangles.size());

  // Describe buffer as array of VertexObj.
  VkAccelerationStructureGeometryTrianglesDataKHR triangles{
      .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
      .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,  // vec3 vertex position data
      .vertexData   = {.deviceAddress = vertexAddress},
      .vertexStride = sizeof(nvh::PrimitiveVertex),
      .maxVertex    = static_cast<uint32_t>(prim.vertices.size()) - 1,
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
  nvh::ScopedTimer st(__FUNCTION__);

  size_t numMeshes = m_meshes.size();

  // BLAS - Storing each primitive in a geometry
  std::vector<nvvk::AccelerationStructureBuildData> blasBuildData(numMeshes);
  m_blas.resize(numMeshes);  // All BLAS

  // Get the build information for all the BLAS
  VkDeviceSize maxScratchSize = 0;
  for(uint32_t p_idx = 0; p_idx < numMeshes; p_idx++)
  {
    const VkDeviceAddress vertex_address = nvvk::getBufferDeviceAddress(m_device, m_bMeshes[p_idx].vertices.buffer);
    const VkDeviceAddress index_address  = nvvk::getBufferDeviceAddress(m_device, m_bMeshes[p_idx].indices.buffer);
    blasBuildData[p_idx].asType          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    blasBuildData[p_idx].addGeometry(primitiveToGeometry(m_meshes[p_idx], vertex_address, index_address));
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo =
        blasBuildData[p_idx].finalizeGeometry(m_device, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
    maxScratchSize = std::max(maxScratchSize, sizeInfo.buildScratchSize);
  }

  // Create the scratch buffer
  nvvk::Buffer scratchBuffer =
      m_alloc->createBuffer(maxScratchSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  // Create the acceleration structures
  VkCommandBuffer cmd = m_app->createTempCmdBuffer();
  {
    for(uint32_t p_idx = 0; p_idx < numMeshes; p_idx++)
    {
      m_blas[p_idx] = m_alloc->createAcceleration(blasBuildData[p_idx].makeCreateInfo());
      blasBuildData[p_idx].cmdBuildAccelerationStructure(cmd, m_blas[p_idx].accel, scratchBuffer.address);
    }
  }
  m_app->submitAndWaitTempCmdBuffer(cmd);

  // Cleanup
  m_alloc->destroy(scratchBuffer);
}

//--------------------------------------------------------------------------------------------------
// Create a regular top level acceleration structures from the scene BLAS'es
//
void PartitionedTlasSample::createTopLevelAS()
{
  nvh::ScopedTimer st(__FUNCTION__);

  std::vector<VkAccelerationStructureInstanceKHR> tlasInstances;
  tlasInstances.resize(m_nodes.size());
  uint32_t numThreads = std::min((uint32_t)m_nodes.size(), std::thread::hardware_concurrency());
  nvh::parallel_batches(
      m_nodes.size(),
      [&](uint64_t i) {
        auto& node = m_nodes[i];

        VkAccelerationStructureInstanceKHR instance{
            .transform           = nvvk::toTransformMatrixKHR(node.localMatrix()),  // Position of the instance
            .instanceCustomIndex = static_cast<uint32_t>(node.mesh),                // gl_InstanceCustomIndexEX
            .mask                = 0xFF,                                            // All objects
            .instanceShaderBindingTableRecordOffset = 0,  // We will use the same hit group for all object
            .flags                                  = {},
            .accelerationStructureReference         = m_blas[node.mesh].address,
        };
        tlasInstances[i] = instance;
      },
      numThreads);

  VkCommandBuffer cmd = m_app->createTempCmdBuffer();

  // Create the instances buffer, add a barrier to ensure the data is copied before the TLAS build
  m_instancesBuffer = m_alloc->createBuffer(cmd, tlasInstances,
                                            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                                                | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);

  m_tlasBuildData.asType = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

  nvvk::AccelerationStructureGeometryInfo geometryInfo =
      m_tlasBuildData.makeInstanceGeometry(tlasInstances.size(), m_instancesBuffer.address);
  m_tlasBuildData.addGeometry(geometryInfo);
  // Get the size of the TLAS
  auto sizeInfo = m_tlasBuildData.finalizeGeometry(m_device, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                                                 | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR);

  if(sizeInfo.accelerationStructureSize == ~VkDeviceSize(0) || sizeInfo.buildScratchSize == ~VkDeviceSize(0))
  {
    LOGE("Error: failed to get the size of the TLAS - Regular TLAS unavailable\n");
    return;
  }

  m_stats.tlasAccelerationStructureSize = sizeInfo.accelerationStructureSize;
  m_stats.tlasBuildScratchSize          = sizeInfo.buildScratchSize;

  // Create the scratch buffer
  m_tlasScratchBuffer = m_alloc->createBuffer(sizeInfo.buildScratchSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                                                             | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  // Create the TLAS
  m_tlas = m_alloc->createAcceleration(m_tlasBuildData.makeCreateInfo());
  m_tlasBuildData.cmdBuildAccelerationStructure(cmd, m_tlas.accel, m_tlasScratchBuffer.address);
  m_app->submitAndWaitTempCmdBuffer(cmd);

  m_alloc->finalizeAndReleaseStaging();
}

// Create a partitions acceleration structure from the scene BLAS'es
void PartitionedTlasSample::createPartitionedTopLevelAS()
{
  nvh::ScopedTimer st(__FUNCTION__);
  // The partitions are defined on a simple grid
  uint32_t partitionCount = m_partitionCountPerAxis * m_partitionCountPerAxis + 1;  // +1 is for the ground plane

  uint32_t maxInstancesPerPartition = 0;


  // Build the instance information for each domino, and add them to their respective partitions
  std::vector<VkPartitionedAccelerationStructureWriteInstanceDataNV> instances;
  std::unordered_map<uint32_t, uint32_t>                              partitionSizes;

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

  uint32_t numThreads = std::min((uint32_t)m_nodes.size(), std::thread::hardware_concurrency());


  nvh::parallel_batches(
      m_nodes.size(),
      [&](uint64_t i) {
        nvh::Node&                                             node = m_nodes[i];
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
      numThreads);


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

  buffers.accelerationStructure =
      m_alloc->createBuffer(sizeInfo.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                                                                    | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

  buffers.buildScratch = m_alloc->createBuffer(sizeInfo.buildScratchSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

  if(sizeInfo.updateScratchSize != 0)
  {
    buffers.updateScratch = m_alloc->createBuffer(sizeInfo.updateScratchSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  }


  buffers.operationsInfo = m_alloc->createBuffer(sizeInfo.operationInfoSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                                                                 | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

  buffers.operationsCount = m_alloc->createBuffer(sizeInfo.operationCountSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                                                                   | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

  buffers.instanceWriteInfo = m_alloc->createBuffer(sizeInfo.instanceWriteInfoSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                                                                        | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

  if(sizeInfo.instanceUpdateInfoSize > 0)
  {
    buffers.instanceUpdateInfo = m_alloc->createBuffer(sizeInfo.instanceUpdateInfoSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                                                                            | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  }

  if(sizeInfo.partitionWriteInfoSize > 0)
  {
    buffers.partitionWriteInfo = m_alloc->createBuffer(sizeInfo.partitionWriteInfoSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                                                                            | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  }


  m_ptlas.setBuffers(buffers);

  m_animationShaderData.originalInstanceWriteInfo = m_ptlas.getBuffers().instanceWriteInfo.address;
  m_animationShaderData.globalPartitionIndex      = partitionCount - 1;

  // Upload the instance and partition data, and perform the first PTLAS build

  {
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();

    m_ptlas.uploadPtlasData(m_alloc.get(), cmd, instances);

    memoryBarrier(cmd);

    m_ptlas.buildAccelerationStructure(cmd);


    m_app->submitAndWaitTempCmdBuffer(cmd);
  }
  m_alloc->releaseStaging();

  // Allocate a buffer to store the dynamic instance updates during animation
  m_partitionedTlasInstanceWriteDynamic =
      m_alloc->createBuffer(sizeInfo.instanceWriteInfoSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

#ifdef USE_NVVK_INSPECTOR
  {
    nvvkhl::ElementInspector::BufferInspectionInfo info{};
    info.entryCount = sizeInfo.partitionWriteInfoSize / sizeof(VkPartitionedAccelerationStructureWritePartitionDataNV);
    info.format     = g_elementInspector->formatStruct(
        "struct VkPartitionedAccelerationStructureWritePartitionDataNV\
  {\
    uint32_t        partitionIndex;\
    vec3           partitionTranslation;\
    uint32_t        instanceCount;\
    uint32_t        instanceIndicesStrideInBytes;\
    uint64_t instanceIndices;\
  }");
    info.name         = "partition write";
    info.sourceBuffer = m_partitionedTlasPartitionWriteDynamicWholeState.buffer;
    g_elementInspector->initBufferInspection(ePartitionWrite, info);
  }

  {
    nvvkhl::ElementInspector::BufferInspectionInfo info{};
    info.entryCount = sizeInfo.partitionWriteInfoSize / sizeof(VkPartitionedAccelerationStructureWritePartitionDataNV);
    info.format     = g_elementInspector->formatStruct(
        "struct VkPartitionedAccelerationStructureWritePartitionDataNV\
  {\
    uint32_t        partitionIndex;\
    vec3           partitionTranslation;\
    uint32_t        instanceCount;\
    uint32_t        instanceIndicesStrideInBytes;\
    uint64_t instanceIndices;\
  }");
    info.name         = "partition write original";
    info.sourceBuffer = m_ptlas.getBuffers().partitionWriteInfo.buffer;
    g_elementInspector->initBufferInspection(ePartitionWriteOriginal, info);
  }


  {
    nvvkhl::ElementInspector::BufferInspectionInfo info{};
    info.entryCount = sizeInfo.instanceWriteInfoSize / sizeof(VkPartitionedAccelerationStructureWriteInstanceDataNV);
    info.format     = g_elementInspector->formatStruct(
        "struct PartitionedAccelerationStructureWriteInstanceDataNV\
    {\
      mat3x4 transform;\
      vec3    explicitAABBMin;\
      vec3    explicitAABBMax;\
      uint32_t instanceIDInstanceMask;\
      uint32_t instanceContributionToHitGroupIndex;\
      uint32_t instanceFlags;\
      uint32_t                                         instanceIndex;\
      uint64_t                                    accelerationStructure;\
    }");
    info.name         = "instance write";
    info.sourceBuffer = m_partitionedTlasInstanceWriteDynamic.buffer;
    g_elementInspector->initBufferInspection(eInstanceWrite, info);
  }

  {
    nvvkhl::ElementInspector::BufferInspectionInfo info{};
    info.entryCount = sizeInfo.instanceWriteInfoSize / sizeof(VkPartitionedAccelerationStructureWriteInstanceDataNV);
    info.format     = g_elementInspector->formatStruct(
        "struct PartitionedAccelerationStructureWriteInstanceDataNV\
    {\
      mat3x4 transform;\
      vec3    explicitAABBMin;\
      vec3    explicitAABBMax;\
      uint32_t instanceIDInstanceMask;\
      uint32_t instanceContributionToHitGroupIndex;\
      uint32_t instanceFlags;\
      uint32_t                                         instanceIndex;\
      uint64_t                                    accelerationStructure;\
    }");
    info.name         = "instance write original";
    info.sourceBuffer = m_ptlas.getBuffers().instanceWriteInfo.buffer;
    g_elementInspector->initBufferInspection(eInstanceWriteOriginal, info);
  }

  {

    nvvkhl::ElementInspector::BufferInspectionInfo info{};
    info.entryCount = 2;
    info.format     = g_elementInspector->formatStruct(
        "struct VkBuildPartitionedAccelerationStructureIndirectCommandNV\
    {\
      uint32_t opType;\
      uint32_t                                    argCount;\
      u64vec2                   argData;\
    }");
    info.name         = "ops";
    info.sourceBuffer = buffers.operationsInfo.buffer;
    g_elementInspector->initBufferInspection(eOps, info);
  }

  {

    nvvkhl::ElementInspector::BufferInspectionInfo info{};
    info.entryCount   = sizeInfo.instanceIndicesSize / sizeof(uint32_t);
    info.format       = g_elementInspector->formatUint32();
    info.name         = "instanceIndices";
    info.sourceBuffer = buffers.instanceIndices.buffer;
    g_elementInspector->initBufferInspection(eInstanceIndices, info);
  }

  {

    nvvkhl::ElementInspector::BufferInspectionInfo info{};
    info.entryCount   = sizeInfo.instanceIndicesSize / sizeof(uint32_t);
    info.format       = g_elementInspector->formatUint32();
    info.name         = "instanceIndices original";
    info.sourceBuffer = m_partitionedTlasInstanceListOriginal.buffer;
    g_elementInspector->initBufferInspection(eInstanceIndicesOriginal, info);
  }


#endif
}
