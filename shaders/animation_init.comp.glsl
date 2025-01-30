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
Initialization of the animation state and partition data
*/
#version 460
#extension GL_GOOGLE_include_directive : enable
#include "animation_common.h"


void main()
{
  const uint32_t globalIndex  = gl_GlobalInvocationID.x;
  const uint32_t dynamicIndex = globalIndexToDynamicIndex(globalIndex, constants);


  // Reset collision tracker for automatic camera movement
  if(globalIndex == 0)
  {
    AnimationGlobalStateBuffer(constants.globalState).s.currentCollisionIndex = 0;
  }

  // For dynamic objects reset their animation state to the original
  if(dynamicIndex != ~0u)
  {
    AnimationStateBuffer(constants.state[(constants.currentStateIndex + 0) % 2]).s[dynamicIndex] =
        AnimationStateBuffer(constants.originalState).s[dynamicIndex];

    AnimationStateBuffer(constants.state[(constants.currentStateIndex + 1) % 2]).s[dynamicIndex] =
        AnimationStateBuffer(constants.originalState).s[dynamicIndex];
  }

  // If PTLAS is active, reset all the PTLAS- and partition-related data
  if(constants.ptlasActive == 1)
  {

    // Rewrite the instance definitions
    if(globalIndex < constants.totalObjectCount)
    {
      PartitionedAccelerationStructureWriteInstanceDataNVBuffer(constants.instanceWriteInfo).s[globalIndex] =
          PartitionedAccelerationStructureWriteInstanceDataNVBuffer(constants.originalInstanceWriteInfo).s[globalIndex];

      // The build will have to rewrite all the instances of the scene
      if(globalIndex == 0)
      {
        BuildPartitionedAccelerationStructureIndirectCommandNVBuffer(constants.ptlasOperations).s[0].argCount =
            constants.totalObjectCount;
        ;
      }
    }
  }
}