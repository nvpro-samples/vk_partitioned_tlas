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

//////////////////////////////////////////////////////////////////////////
//
// # GL_NV_partitioned_acceleration_structure
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

#ifdef __cplusplus
#include <cstdint>
#endif

#ifndef VK_NV_partitioned_acceleration_structure
#define VK_NV_partitioned_acceleration_structure 1
#define VK_NV_PARTITIONED_ACCELERATION_STRUCTURE_SPEC_VERSION 1

#define VK_PARTITIONED_ACCELERATION_STRUCTURE_PARTITION_INDEX_GLOBAL_NV (~0U)

#define DeviceAddress uint64_t
#define DeviceSize uint64_t

struct StridedDeviceAddressNV
{
  DeviceAddress startAddress;
  DeviceSize    strideInBytes;
};


#define VK_PARTITIONED_ACCELERATION_STRUCTURE_OP_TYPE_WRITE_INSTANCE_NV 0
#define VK_PARTITIONED_ACCELERATION_STRUCTURE_OP_TYPE_UPDATE_INSTANCE_NV 1
#define VK_PARTITIONED_ACCELERATION_STRUCTURE_OP_TYPE_WRITE_PARTITION_NV 2
#define VK_PARTITIONED_ACCELERATION_STRUCTURE_OP_TYPE_MAX_ENUM_NV 0x7FFFFFFF
#define PartitionedAccelerationStructureOpTypeNV uint32_t

struct BuildPartitionedAccelerationStructureIndirectCommandNV
{
  PartitionedAccelerationStructureOpTypeNV opType;
  uint32_t                                  argCount;
  StridedDeviceAddressNV                   argData;
};

#define PartitionedAccelerationStructureInstanceFlagsNV uint32_t

#define VK_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCE_FLAG_TRIANGLE_FACING_CULL_DISABLE_BIT_NV 0x00000001
#define VK_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCE_FLAG_TRIANGLE_FLIP_FACING_BIT_NV 0x00000002
#define VK_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCE_FLAG_FORCE_OPAQUE_BIT_NV 0x00000004
#define VK_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCE_FLAG_FORCE_NO_OPAQUE_BIT_NV 0x00000008
#define VK_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCE_FLAG_ENABLE_EXPLICIT_BOUNDING_BOX_NV 0x00000010
#define VK_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCE_FLAG_BITS_MAX_ENUM_NV 0x7FFFFFFF

#define TransformMatrixKHR mat3x4
struct PartitionedAccelerationStructureWriteInstanceDataNV
{
  TransformMatrixKHR                               transform;
  float                                              explicitAABB[6];
  uint32_t                                           instanceID;
  uint32_t                                           instanceMask;
  uint32_t                                           instanceContributionToHitGroupIndex;
  PartitionedAccelerationStructureInstanceFlagsNV instanceFlags;
  uint32_t                                           instanceIndex;
  uint32_t                                           partitionIndex;
  DeviceAddress                                    accelerationStructure;
};

struct PartitionedAccelerationStructureUpdateInstanceDataNV
{
  uint32_t      instanceIndex;
  uint32_t      instanceContributionToHitGroupIndex;
  DeviceAddress accelerationStructure;
};

struct PartitionedAccelerationStructureWritePartitionDataNV
{
  uint32_t partitionIndex;
  float    partitionTranslation[3];
};

#endif