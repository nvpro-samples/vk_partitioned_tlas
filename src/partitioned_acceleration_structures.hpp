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
#pragma once

#include "nvvk/buffers_vk.hpp"
#include "nvvk/resourceallocator_vk.hpp"
#include "vulkan/vulkan_core.h"
#include <sstream>

namespace nvvk {

// Helper class for Partitioned Acceleration Structures (PTLAS)
// This class provides estimates for all the memory required to generate, store and maintain
// a PTLAS, including the device-side buffers containing instance and partition data
// This class also provides the uploadPtlasData function to initialize the contents of those
// buffers and upload them on the device
class PartitionedAccelerationStructure
{
public:
  void init(VkDevice device) { m_device = device; }

  void deinit() { *this = {}; }

  // PTLAS description used to query the build sizes
  struct BuildSizeQuery
  {
    // Total number of instances
    uint32_t instanceCount;
    // Total number of partitions
    uint32_t partitionCount;
    // Maximum number of instances in any given partition
    uint32_t maxInstancesPerPartition;
    // Maximum number of instances in the global partition, identified with index ~0u. This partition is typically used to put dynamic objects that cannot be easily organized spatially
    uint32_t maxInstancesInGlobalPartition;
    // Maximum number of simultaneous operations to perform on the AS. Typically 2: 1 for for writing instance data, 1 for writing partition data
    uint32_t maxOperations;
    // Build flags for the acceleration structure
    VkBuildAccelerationStructureFlagsKHR buildFlags = {};
    // If true, allow a fast path for updating the BLAS underlying the instances
    bool allowInstanceUpdate{false};
    // If true, allow using per-partition translation vectors
    bool allowPartitionTranslation{false};
    // Flags for the PTLAS
    VkPartitionedAccelerationStructureFlagsNV ptlasFlags = {VK_STRUCTURE_TYPE_PARTITIONED_ACCELERATION_STRUCTURE_FLAGS_NV};
  };


  // Size information for the PTLAS buffers
  struct BuildSizeInfo
  {
    // Buffers for the actual build:
    //   Acceleration structure storage
    VkDeviceSize accelerationStructureSize;
    //   Scratch memory used for AS build
    VkDeviceSize buildScratchSize;
    //   (Optional) Scratch memory used for AS update
    VkDeviceSize updateScratchSize;

    // Auxiliary buffers holding build operations, instance and partition info:
    //   Operations to perform on the AS (write, update of instances and partitions)
    VkDeviceSize operationInfoSize;
    //   Number of operations to perform
    VkDeviceSize operationCountSize;
    //   Instance information to write into the AS
    VkDeviceSize instanceWriteInfoSize;
    //   (Optional) Instance update information to update the BLAS of the instances
    VkDeviceSize instanceUpdateInfoSize;
    //   (Optional) Partition information to write into the AS
    VkDeviceSize partitionWriteInfoSize;
  };


  // Storage of the PTLAS along with its auxiliary buffers
  struct Buffers
  {
    // Buffers for the actual build:
    //   Actual acceleration structure storage
    nvvk::Buffer accelerationStructure;
    //   Scratch memory used for AS build
    nvvk::Buffer buildScratch;
    //   (Optional) Scratch memory used for AS update
    nvvk::Buffer updateScratch;

    // Auxiliary buffers holding build operations, instance and partition info:
    //   Operations to perform on the AS (write, update of instances and partitions)
    nvvk::Buffer operationsInfo;
    //   Number of operations to perform
    nvvk::Buffer operationsCount;
    //   Instance information to write into the AS
    nvvk::Buffer instanceWriteInfo;
    //   (Optional) Instance update information to update the BLAS of the instances
    nvvk::Buffer instanceUpdateInfo;
    //   (Optional) Partition information to write into the AS
    nvvk::Buffer partitionWriteInfo;
  };

  // Partition definition for initial upload
  struct PartitionWrite
  {
    // Unique index of the partition
    uint32_t partitionIndex;
    // Indices of the instances belonging to that partition in the array containing all instances
    std::vector<uint32_t> instanceIndices;
  };

  // Get the sizes of the buffers for the actual build as well as auxiliary buffers storing build operations, instance and partition info
  BuildSizeInfo getBuildSizes(const BuildSizeQuery& buildSizeQuery)
  {

    BuildSizeInfo buildSizes{};

    m_ptlasFlags = buildSizeQuery.ptlasFlags;

    m_inputInfo                = {VK_STRUCTURE_TYPE_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCES_INPUT_NV};
    m_inputInfo.flags          = buildSizeQuery.buildFlags;
    m_inputInfo.instanceCount  = buildSizeQuery.instanceCount;
    m_inputInfo.partitionCount = buildSizeQuery.partitionCount;
    m_inputInfo.maxInstancePerPartitionCount      = buildSizeQuery.maxInstancesPerPartition;
    m_inputInfo.maxInstanceInGlobalPartitionCount = buildSizeQuery.maxInstancesInGlobalPartition;
    m_inputInfo.pNext                             = &m_ptlasFlags;

    VkAccelerationStructureBuildSizesInfoKHR ptlasSizeInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetPartitionedAccelerationStructuresBuildSizesNV(m_device, &m_inputInfo, &ptlasSizeInfo);

    buildSizes.accelerationStructureSize = ptlasSizeInfo.accelerationStructureSize;

    buildSizes.buildScratchSize  = ptlasSizeInfo.buildScratchSize;
    buildSizes.updateScratchSize = ptlasSizeInfo.updateScratchSize;
    buildSizes.operationInfoSize = buildSizeQuery.maxOperations * sizeof(VkBuildPartitionedAccelerationStructureIndirectCommandNV);
    buildSizes.operationCountSize = sizeof(uint32_t);
    buildSizes.instanceWriteInfoSize = buildSizeQuery.instanceCount * sizeof(VkPartitionedAccelerationStructureWriteInstanceDataNV);
    if(buildSizeQuery.allowInstanceUpdate)
    {
      buildSizes.instanceUpdateInfoSize =
          buildSizeQuery.instanceCount * sizeof(VkPartitionedAccelerationStructureUpdateInstanceDataNV);
    }
    if(buildSizeQuery.allowPartitionTranslation)
    {
      buildSizes.partitionWriteInfoSize =
          (buildSizeQuery.partitionCount + 1) * sizeof(VkPartitionedAccelerationStructureWritePartitionTranslationDataNV);
    }
    return buildSizes;
  }


  void setBuffers(const Buffers& buffers) { m_buffers = buffers; }


  const Buffers& getBuffers() const { return m_buffers; }

  // Write instance and partition data from host-side storage. Each partition is defined by an entry in partitionData, with its set of instances. The partition ID is directly deduced from the index in partitionData.
  void uploadPtlasData(nvvk::ResourceAllocator* alloc,
                       VkCommandBuffer          cmd,
                       const std::vector<VkPartitionedAccelerationStructureWriteInstanceDataNV>& instances,  // All instances to be stored in the PTLAS
                       const std::vector<VkPartitionedAccelerationStructureWritePartitionTranslationDataNV>& partitions = {}  // (Optional) Per-partition translation vectors
  )

  {
    uint32_t instanceCount = m_inputInfo.instanceCount;


    uploadBuffer(alloc, cmd, m_buffers.instanceWriteInfo, instances);

    if(!partitions.empty())
    {
      uploadBuffer(alloc, cmd, m_buffers.partitionWriteInfo, partitions);
    }

    uint32_t operationCount = partitions.empty() ? 1 : 2;
    // To finalize the upload the PTLAS instances and partitions we need up to 2 operations, one to write the instances, another to write the optional per-partition translations vectors (see below)
    vkCmdUpdateBuffer(cmd, m_buffers.operationsCount.buffer, 0, sizeof(uint32_t), &operationCount);

    // PTLAS build operations
    {
      std::vector<VkBuildPartitionedAccelerationStructureIndirectCommandNV> srcOperations(operationCount);

      // Instance write
      srcOperations[0]                       = {};
      srcOperations[0].opType                = VK_PARTITIONED_ACCELERATION_STRUCTURE_OP_TYPE_WRITE_INSTANCE_NV;
      srcOperations[0].argCount              = instanceCount;
      srcOperations[0].argData.startAddress  = m_buffers.instanceWriteInfo.address;
      srcOperations[0].argData.strideInBytes = sizeof(VkPartitionedAccelerationStructureWriteInstanceDataNV);

      if(operationCount > 1)
      {
        // Partition write
        srcOperations[1]                       = {};
        srcOperations[1].opType                = VK_PARTITIONED_ACCELERATION_STRUCTURE_OP_TYPE_WRITE_PARTITION_TRANSLATION_NV;
        srcOperations[1].argCount              = uint32_t(partitions.size());
        srcOperations[1].argData.startAddress  = m_buffers.partitionWriteInfo.address;
        srcOperations[1].argData.strideInBytes = sizeof(VkPartitionedAccelerationStructureWritePartitionTranslationDataNV);
      }
      uploadBuffer(alloc, cmd, m_buffers.operationsInfo, srcOperations);
    }
  }

  // Build/Update the PTLAS, either from scratch or from the existing PTLAS
  void buildAccelerationStructure(VkCommandBuffer cmd, bool update = false)
  {
    VkBuildPartitionedAccelerationStructureInfoNV commandInfo = {VK_STRUCTURE_TYPE_BUILD_PARTITIONED_ACCELERATION_STRUCTURE_INFO_NV};
    commandInfo.input = m_inputInfo;
    if(update)
    {
      commandInfo.srcAccelerationStructureData = m_buffers.accelerationStructure.address;
    }
    else
    {
      commandInfo.srcAccelerationStructureData = 0;
    }
    commandInfo.dstAccelerationStructureData = m_buffers.accelerationStructure.address;
    commandInfo.scratchData                  = m_buffers.buildScratch.address;
    commandInfo.srcInfos                     = m_buffers.operationsInfo.address;
    commandInfo.srcInfosCount                = m_buffers.operationsCount.address;

    vkCmdBuildPartitionedAccelerationStructuresNV(cmd, &commandInfo);
  }

  // Get the VkWriteDescriptorSet with appropriate pNext to set bind the acceleration structure
  VkWriteDescriptorSet getWriteDescriptorSet(VkDescriptorSet dset, uint32_t bindingPoint)
  {
    m_writeDescriptorSetExtension.accelerationStructureCount = 1;
    m_writeDescriptorSetExtension.pAccelerationStructures    = &m_buffers.accelerationStructure.address;

    VkWriteDescriptorSet writeDescriptorSet = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writeDescriptorSet.descriptorCount      = 1;
    writeDescriptorSet.dstArrayElement      = 0;
    writeDescriptorSet.dstBinding           = bindingPoint;
    writeDescriptorSet.dstSet               = dset;

    writeDescriptorSet.pNext          = &m_writeDescriptorSetExtension;
    writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_PARTITIONED_ACCELERATION_STRUCTURE_NV;
    return writeDescriptorSet;
  }

private:
  template <typename T>
  void uploadBuffer(nvvk::ResourceAllocator* alloc, VkCommandBuffer cmd, const nvvk::Buffer& dst, const std::vector<T>& src)
  {
    if(src.size() > 0 && dst.buffer != VK_NULL_HANDLE)
    {
      alloc->getStaging()->cmdToBuffer(cmd, dst.buffer, 0, src.size() * sizeof(T), src.data());
    }
  }

  // Storage buffers for acceleration structure and auxiliary data
  Buffers m_buffers;

  // Instance and partition info for AS build
  VkPartitionedAccelerationStructureInstancesInputNV m_inputInfo{VK_STRUCTURE_TYPE_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCES_INPUT_NV};

  VkPartitionedAccelerationStructureFlagsNV m_ptlasFlags = {VK_STRUCTURE_TYPE_PARTITIONED_ACCELERATION_STRUCTURE_FLAGS_NV};

  // PTLAS-specific info to be put in the pNext pointer of the VkWriteDescriptorSet
  VkWriteDescriptorSetPartitionedAccelerationStructureNV m_writeDescriptorSetExtension{
      VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_PARTITIONED_ACCELERATION_STRUCTURE_NV};

  VkDevice m_device{};
};

}  // namespace nvvk