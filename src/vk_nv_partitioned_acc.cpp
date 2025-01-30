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

#include "vk_nv_partitioned_acc.h"
#include <nvvk/extensions_vk.hpp>

static PFN_vkGetPartitionedAccelerationStructuresBuildSizesNV s_vkGetPartitionedAccelerationStructuresBuildSizesNV = nullptr;
static PFN_vkCmdBuildPartitionedAccelerationStructuresNV s_vkCmdBuildPartitionedAccelerationStructuresNV = nullptr;

#ifndef NVVK_HAS_VK_NV_partitioned_accleration_structure
VKAPI_ATTR void VKAPI_CALL vkGetPartitionedAccelerationStructuresBuildSizesNV(VkDevice device,
                                                                               const VkPartitionedAccelerationStructureInstancesInputNV* info,
                                                                               VkAccelerationStructureBuildSizesInfoKHR* pSizeInfo)
{
  s_vkGetPartitionedAccelerationStructuresBuildSizesNV(device, info, pSizeInfo);
}

VKAPI_ATTR void VKAPI_CALL vkCmdBuildPartitionedAccelerationStructuresNV(VkCommandBuffer cmd,
                                                                          const VkBuildPartitionedAccelerationStructureInfoNV* buildInfo)
{
  s_vkCmdBuildPartitionedAccelerationStructuresNV(cmd, buildInfo);
}
#endif

VkBool32 load_VK_NV_partitioned_accleration_structure(VkInstance instance, VkDevice device)
{
  s_vkGetPartitionedAccelerationStructuresBuildSizesNV = nullptr;
  s_vkCmdBuildPartitionedAccelerationStructuresNV      = nullptr;

  s_vkGetPartitionedAccelerationStructuresBuildSizesNV =
      (PFN_vkGetPartitionedAccelerationStructuresBuildSizesNV)vkGetDeviceProcAddr(device, "vkGetPartitionedAccelerationStructuresBuildSizesNV");
  s_vkCmdBuildPartitionedAccelerationStructuresNV =
      (PFN_vkCmdBuildPartitionedAccelerationStructuresNV)vkGetDeviceProcAddr(device, "vkCmdBuildPartitionedAccelerationStructuresNV");

  return s_vkGetPartitionedAccelerationStructuresBuildSizesNV && s_vkCmdBuildPartitionedAccelerationStructuresNV;
}
