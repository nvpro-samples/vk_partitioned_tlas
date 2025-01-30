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

#include <vulkan/vulkan_core.h>

//////////////////////////////////////////////////////////////////////////
//
// # VK_NV_partitioned_acceleration_structure
//
// Partitions divide a fixed pool with a maximum size of number of instances across a top-level
// acceleration structure (AS). The feature is also referred to as "PTLAS".
//
// # Common
//
// Both new extensions are "multi indirect", however with slightly different designs.
// Cluster builds are one type of operation per single commandbuffer command, following
// the traditional indirect approach.
//
// Partition builds/updates use two level-indirection, meaning multiple operation types
// can be executed per single commandbuffer command, and the types are also sourced
// from GPU

#ifndef VK_NV_partitioned_acceleration_structure
#define VK_NV_partitioned_acceleration_structure 1
#define VK_NV_PARTITIONED_ACCELERATION_STRUCTURE_SPEC_VERSION 1
#define VK_NV_PARTITIONED_ACCELERATION_STRUCTURE_EXTENSION_NAME "VK_NV_partitioned_acceleration_structure"
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PARTITIONED_ACCELERATION_STRUCTURE_FEATURES_NV ((VkStructureType)1000570000)
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PARTITIONED_ACCELERATION_STRUCTURE_PROPERTIES_NV ((VkStructureType)1000570001)
#define VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_PARTITIONED_ACCELERATION_STRUCTURE_NV ((VkStructureType)1000570002)
#define VK_STRUCTURE_TYPE_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCES_INPUT_NV ((VkStructureType)1000570003)
#define VK_STRUCTURE_TYPE_BUILD_PARTITIONED_ACCELERATION_STRUCTURE_INFO_NV ((VkStructureType)1000570004)
#define VK_STRUCTURE_TYPE_PARTITIONED_ACCELERATION_STRUCTURE_FLAGS_NV ((VkStructureType)1000570005)
#define VK_DESCRIPTOR_TYPE_PARTITIONED_ACCELERATION_STRUCTURE_NV ((VkDescriptorType)1000570000)
#define VK_PARTITIONED_ACCELERATION_STRUCTURE_PARTITION_INDEX_GLOBAL_NV (~0U)

typedef struct VkStridedDeviceAddressNV
{
  VkDeviceAddress startAddress;
  VkDeviceSize    strideInBytes;
} VkStridedDeviceAddressNV;


typedef struct VkPhysicalDevicePartitionedAccelerationStructureFeaturesNV
{
  VkStructureType sType;
  void*           pNext;
  VkBool32        partitionedAccelerationStructure;
} VkPhysicalDevicePartitionedAccelerationStructureFeaturesNV;

typedef struct VkPhysicalDevicePartitionedAccelerationStructurePropertiesNV
{
  VkStructureType sType;
  void*           pNext;
  uint32_t        maxPartitionCount;
} VkPhysicalDevicePartitionedAccelerationStructurePropertiesNV;

typedef struct VkPartitionedAccelerationStructureFlagsNV
{
  VkStructureType sType;
  void*           pNext;
  VkBool32        enablePartitionTranslation;
} VkPartitionedAccelerationStructureFlagsNV;

typedef enum VkPartitionedAccelerationStructureOpTypeNV
{
  VK_PARTITIONED_ACCELERATION_STRUCTURE_OP_TYPE_WRITE_INSTANCE_NV  = 0,
  VK_PARTITIONED_ACCELERATION_STRUCTURE_OP_TYPE_UPDATE_INSTANCE_NV = 1,
  VK_PARTITIONED_ACCELERATION_STRUCTURE_OP_TYPE_WRITE_PARTITION_NV = 2,
  VK_PARTITIONED_ACCELERATION_STRUCTURE_OP_TYPE_MAX_ENUM_NV        = 0x7FFFFFFF
} VkPartitionedAccelerationStructureOpTypeNV;

typedef struct VkBuildPartitionedAccelerationStructureIndirectCommandNV
{
  VkPartitionedAccelerationStructureOpTypeNV opType;
  uint32_t                                   argCount;
  VkStridedDeviceAddressNV                   argData;
} VkBuildPartitionedAccelerationStructureIndirectCommandNV;

typedef VkFlags VkPartitionedAccelerationStructureInstanceFlagsNV;

typedef enum VkPartitionedAccelerationStructureInstanceFlagBitsNV
{
  VK_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCE_FLAG_TRIANGLE_FACING_CULL_DISABLE_BIT_NV = 0x00000001,
  VK_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCE_FLAG_TRIANGLE_FLIP_FACING_BIT_NV         = 0x00000002,
  VK_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCE_FLAG_FORCE_OPAQUE_BIT_NV                 = 0x00000004,
  VK_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCE_FLAG_FORCE_NO_OPAQUE_BIT_NV              = 0x00000008,
  VK_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCE_FLAG_ENABLE_EXPLICIT_BOUNDING_BOX_NV     = 0x00000010,
  VK_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCE_FLAG_BITS_MAX_ENUM_NV                    = 0x7FFFFFFF
} VkPartitionedAccelerationStructureInstanceFlagBitsNV;

typedef struct VkPartitionedAccelerationStructureWriteInstanceDataNV
{
  VkTransformMatrixKHR                              transform;
  float                                             explicitAABB[6];
  uint32_t                                          instanceID;
  uint32_t                                          instanceMask;
  uint32_t                                          instanceContributionToHitGroupIndex;
  VkPartitionedAccelerationStructureInstanceFlagsNV instanceFlags;
  uint32_t                                          instanceIndex;
  uint32_t                                          partitionIndex;
  VkDeviceAddress                                   accelerationStructure;
} VkPartitionedAccelerationStructureWriteInstanceDataNV;

typedef struct VkPartitionedAccelerationStructureUpdateInstanceDataNV
{
  uint32_t        instanceIndex;
  uint32_t        instanceContributionToHitGroupIndex;
  VkDeviceAddress accelerationStructure;
} VkPartitionedAccelerationStructureUpdateInstanceDataNV;

typedef struct VkPartitionedAccelerationStructureWritePartitionDataNV
{
  uint32_t partitionIndex;
  float    partitionTranslation[3];
} VkPartitionedAccelerationStructureWritePartitionDataNV;

typedef struct VkWriteDescriptorSetPartitionedAccelerationStructureNV
{
  VkStructureType        sType;
  void*                  pNext;
  uint32_t               accelerationStructureCount;
  VkDeviceAddress const* pAccelerationStructures;
} VkWriteDescriptorSetPartitionedAccelerationStructureNV;

typedef struct VkPartitionedAccelerationStructureInstancesInputNV
{
  VkStructureType                      sType;
  void*                                pNext;
  VkBuildAccelerationStructureFlagsKHR flags;
  uint32_t                             instanceCount;
  uint32_t                             maxInstancePerPartitionCount;
  uint32_t                             partitionCount;
  uint32_t                             maxInstanceInGlobalPartitionCount;
} VkPartitionedAccelerationStructureInstancesInputNV;

typedef struct VkBuildPartitionedAccelerationStructureInfoNV
{
  VkStructureType                                    sType;
  void*                                              pNext;
  VkPartitionedAccelerationStructureInstancesInputNV input;
  VkDeviceAddress                                    srcAccelerationStructureData;
  VkDeviceAddress                                    dstAccelerationStructureData;
  VkDeviceAddress                                    scratchData;
  VkDeviceAddress                                    srcInfos;
  VkDeviceAddress                                    srcInfosCount;
} VkBuildPartitionedAccelerationStructureInfoNV;

typedef void(VKAPI_PTR* PFN_vkGetPartitionedAccelerationStructuresBuildSizesNV)(VkDevice device,
                                                                                const VkPartitionedAccelerationStructureInstancesInputNV* pInfo,
                                                                                VkAccelerationStructureBuildSizesInfoKHR* pBuildInfo);
typedef void(VKAPI_PTR* PFN_vkCmdBuildPartitionedAccelerationStructuresNV)(VkCommandBuffer commandBuffer,
                                                                           const VkBuildPartitionedAccelerationStructureInfoNV* pBuildInfo);

#ifndef VK_NO_PROTOTYPES
VKAPI_ATTR void VKAPI_CALL vkGetPartitionedAccelerationStructuresBuildSizesNV(VkDevice device,
                                                                              VkPartitionedAccelerationStructureInstancesInputNV const* pInfo,
                                                                              VkAccelerationStructureBuildSizesInfoKHR* pBuildInfo);

VKAPI_ATTR void VKAPI_CALL vkCmdBuildPartitionedAccelerationStructuresNV(VkCommandBuffer commandBuffer,
                                                                         VkBuildPartitionedAccelerationStructureInfoNV const* pBuildInfo);
#endif

VkBool32 load_VK_NV_partitioned_accleration_structure(VkInstance instance, VkDevice device);

#endif