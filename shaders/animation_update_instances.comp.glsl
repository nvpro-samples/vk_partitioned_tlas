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

Update the update of the PTLAS instance data and per-partition instance indices

*/
#version 460
#extension GL_GOOGLE_include_directive : enable
#include "animation_common.h"


void main()
{
  // Fetch the index of the domino within the domino list
  uint32_t dynamicIndex = gl_GlobalInvocationID.x < constants.dynamicObjectCount ? gl_GlobalInvocationID.x : ~0u;
  // Fetch the index of the domino in the entire scene
  uint32_t globalIndex = dynamicIndexToGlobalIndex(dynamicIndex, constants);


  if(dynamicIndex != ~0u)
  {
    // Fetch the current state of the domino
    AnimationState currDomino = AnimationStateBuffer(constants.state[(constants.currentStateIndex + 1) % 2]).s[dynamicIndex];

    // Check if it has been updated within the last frame
    bool hasMoved = currDomino.lastModified == constants.frameIndex || currDomino.partitionID != currDomino.newPartitionID;
    // If dominoes are moved to the global partition on a per-partition granularity, check if any domino within the domino's partition has been moved within the last frame
    bool forcedFullPartitionMove =
        (constants.dynamicUpdateMode == PTLAS_DYNAMIC_MOVE_TO_GLOBAL)
        && (PartitionStateBuffer(constants.partitionState).s[currDomino.partitionID].lastModified == constants.frameIndex)
        && (constants.dynamicMarkAllDominos != 0);

    // If needed, mark the domino for move into the global partition
    if(forcedFullPartitionMove)
    {
      if(currDomino.partitionID != constants.globalPartitionIndex)
      {
        currDomino.newPartitionID = constants.globalPartitionIndex;
      }
      else
      {
        forcedFullPartitionMove = false;
      }
    }

    // If the domino has moved or needs to be moved to the global partition, update the instance data
    if(hasMoved || forcedFullPartitionMove)
    {
      // Add an entry in the list of modified instances
      uint32_t updateSlot =
          atomicAdd(BuildPartitionedAccelerationStructureIndirectCommandNVBuffer(constants.ptlasOperations).s[0].argCount, 1);

      // Write the updated instance data
      PartitionedAccelerationStructureWriteInstanceDataNVBuffer(constants.instanceWriteInfo).s[updateSlot] =
          PartitionedAccelerationStructureWriteInstanceDataNVBuffer(constants.originalInstanceWriteInfo).s[globalIndex];

      PartitionedAccelerationStructureWriteInstanceDataNVBuffer(constants.instanceWriteInfo).s[updateSlot].transform =
          getTransform3x4(currDomino);

      // If a partition change has been requested, reset the partition change indices
      if(currDomino.partitionID != currDomino.newPartitionID)
      {
        currDomino.partitionID = currDomino.newPartitionID;
        AnimationStateBuffer(constants.state[(constants.currentStateIndex + 1) % 2]).s[dynamicIndex].partitionID =
            currDomino.newPartitionID;
      }
    }

  }
}