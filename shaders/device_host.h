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

#ifndef HOST_DEVICE_H
#define HOST_DEVICE_H

// Uncomment to use the nvvk InspectorElement class for realtime debugging
// Since this involves memory readbacks, the performance hit can be significant when
// using large domino boards
//#define USE_NVVK_INSPECTOR

// If defined in combination with USE_NVVK_INSPECTOR, the Inspector will only display the PTLAS operations buffer
// with minimal overhead. This is useful for debugging the PTLAS update operations.
//#define NVVK_INSPECTOR_OPS_ONLY


#ifndef __cplusplus
#include "gl_nv_partitioned_acc.h"
#extension GL_EXT_control_flow_attributes2 : enable
#else
#include <string>
#include <glm/glm.hpp>
using namespace glm;
#endif

struct FrameInfo
{
  mat4 projInv;
  mat4 viewInv;
};


struct Vertex
{
  vec3 position;
  vec3 normal;
  vec2 texCoord;
};


struct InstanceInfo
{
  mat4 transform;
  int  materialID;
  int  meshID;
};


#define COMPOSITING_SHADER_BLOCK_SIZE 256
struct CompositingShaderData
{
  uint32_t windowSize;
  uint32_t passId;
  uint32_t frameIndex;
};


#define ANIMATION_SHADER_BLOCK_SIZE 256

// Valid values for AnimationShaderData::dynamicUpdateMode
// Update location of dynamic instances within their partition, triggering an update of the related partitions
#define PTLAS_DYNAMIC_ALWAYS_UPDATE 0
// Move dynamic instances to the global partition, bringing them back to their original partition after they become static again
#define PTLAS_DYNAMIC_MOVE_TO_GLOBAL 1
// Use DYNAMIC_ALWAYS_UPDATE for instance in nearby partitions, and DYNAMIC_MOVE_TO_GLOBAL for the farther ones
#define PTLAS_DYNAMIC_UPDATE_OR_MOVE_TO_GLOBAL 2

struct AnimationShaderData
{
  // Address of the regular TLAS instance definitions
  uint64_t instancesAddress;
  // Address of the original state of the dominoes
  uint64_t originalState;

  // Double-buffered dynamic state of the dominoes
  uint64_t state[2];

  // Minimum bbox of the scene
  vec3 objectBboxMin;
  // If != 0, reset the dominoes to their original state
  int resetToOriginal;

  // Maximum bbox of the scene
  vec3 objectBboxMax;
  // Strength of the force applied when right-clicking a domino
  float toppleStrength;

  // Number of dominoes in the scene
  uint32_t dynamicObjectCount;
  // Index of the currently used state buffer
  uint32_t currentStateIndex;
  // Time step between two frames
  float    timeStep;
  uint32_t _pad0;

  // Address of the global state of the animation
  uint64_t globalState;
  // Number of partitions per X and Z axis (regular grid)
  uint32_t partitionsPerAxis;
  // Index of the current frame for animation
  uint32_t frameIndex;

  // Address of the partition state buffer
  uint64_t partitionState;
  // Current mouse coordinates, used to topple dominoes in the closest-hit shader
  ivec2 mouseCoord;

  // Original instance write info, used to reset the dominoes in their original state
  uint64_t originalInstanceWriteInfo;
  // Address of the instance write info buffer used for dynamic updates
  uint64_t instanceWriteInfo;

  // Address of the buffer containing the PTLAS update commands (instance and partition rewrites)
  uint64_t ptlasOperations;
  // Index of the domino to topple in the next animation frame
  int toppleDomino;
  // If != 0, the PTLAS is active, regular TLAS otherwise
  int ptlasActive;

  // Number of physics simulation steps per render frame
  uint32_t subframeCount;
  // Total number of partitions
  int partitionCount;

  // Index of the global partition definition in the buffer of partitions (partitionWriteInfoWholeState)
  uint32_t globalPartitionIndex;
  // Total number of objects in the scene
  uint32_t totalObjectCount;
  // One of PTLAS_DYNAMIC_ALWAYS_UPDATE, PTLAS_DYNAMIC_MOVE_TO_GLOBAL or PTLAS_DYNAMIC_UPDATE_OR_MOVE_TO_GLOBAL
  int32_t dynamicUpdateMode;
  // If != 0, when a domino becomes dynamic all dominoes within the same partition will be marked dynamic as well.
  // In combination with PTLAS_DYNAMIC_MOVE_TO_GLOBAL or PTLAS_DYNAMIC_UPDATE_OR_MOVE_TO_GLOBAL, this reduces the number of updated partitions
  uint32_t dynamicMarkAllDominos;

  // Camera position, used by the PTLAS_DYNAMIC_UPDATE_OR_MOVE_TO_GLOBAL mode to determine the distance between the camera and the partitions
  vec3 eyePosition;
  // In combination with PTLAS_DYNAMIC_UPDATE_OR_MOVE_TO_GLOBAL, defines the distance from which the behavior switches from DYNAMIC_ALWAYS_UPDATE to DYNAMIC_MOVE_TO_GLOBAL
  float dynamicDistanceThreshold;
};

// Domino states for physics simulation
// Domino is in free flight
#define STATE_FREE 0
// Domino is in contact with the ground
#define STATE_GROUND_COLLISION (1u << 0)
// Domino is in contact with another domino
#define STATE_OTHER_COLLISION (1u << 1)
// Domino is toppled by the user
#define STATE_FORCE_TOPPLE (1u << 2)
// Domino is close to the ground, used to damp the physics simulation
#define STATE_CLOSE_TO_GROUND (1u << 3)

// Index of the global partition to use when defining the global partition in the PTLAS update. This is different from AnimationShaderData.globalPartitionIndex
// which is the index of the global partition in the list of partitions stored by the application
#define GLOBAL_PARTITION_INDEX ~0u

#ifndef __cplusplus

void setCollisionState(inout uint32_t v, uint32_t s)
{
  v = (v & ~0x3) | s;
}

bool hasFlag(uint32_t v, uint32_t flag)
{
  return (v & flag) == flag;
}

void addFlag(inout uint32_t v, uint32_t flag)
{
  v = v | flag;
}
void removeFlag(inout uint32_t v, uint32_t flag)
{
  v = v & ~flag;
}
#endif

// Macro to generate a function that returns a string representation of a struct, which can be used for
// the definition of data formats in nvvk::InspectorElement
#ifdef __cplusplus
#define INSPECTABLE_STRUCT(name_, def_)                                                                                \
  struct name_       def_;                                                                                             \
  inline std::string getInspectorString##name_()                                                                       \
  {                                                                                                                    \
    return #def_;                                                                                                      \
  }
#else
#define INSPECTABLE_STRUCT(name_, def_) struct name_ def_;
#endif
INSPECTABLE_STRUCT(MyStruct, {
  uint32_t x;
  uint32_t y;
});

// Domino state for physics simulation
INSPECTABLE_STRUCT(AnimationState, {
  mat3x4 transform3x4;

  vec3     linearVelocity;
  uint32_t stateID;

  vec3     angularVelocity;
  uint32_t partitionID;

  uint32_t lastContact;
  uint32_t firstContact;
  uint32_t lastModified;
  uint32_t newPartitionID;
});


struct AnimationGlobalState
{
  uint32_t currentCollisionIndex;
  uint32_t toppleRequest;
  uint32_t focus;
  uint32_t instanceUpdateCount;
};


#ifndef __cplusplus
void setTransform4x4(inout AnimationState s, mat4 m)
{
  s.transform3x4 = mat3x4(transpose(m));
}

vec3 getPosition(AnimationState s)
{
  return vec3(s.transform3x4[0].w, s.transform3x4[1].w, s.transform3x4[2].w);
}

// Function to set position in transformation matrix
void setPosition(inout AnimationState s, vec3 position)
{
  s.transform3x4[0].w = position.x;
  s.transform3x4[1].w = position.y;
  s.transform3x4[2].w = position.z;
}

float getPositionY(AnimationState s)
{
  return s.transform3x4[1].w;
}

// Function to set position in transformation matrix
void setPositionY(inout AnimationState s, float y)
{
  s.transform3x4[1].w = y;
}


mat4 getTransform4x4(AnimationState s)
{
  return mat4(transpose(s.transform3x4));
}
mat3x4 getTransform3x4(AnimationState s)
{
  return s.transform3x4;
}

void setTransform3x4(inout AnimationState s, mat3x4 m)
{
  s.transform3x4 = m;
}


mat3 getTransform3x3(AnimationState s)
{
  return transpose(mat3(s.transform3x4));
}

vec3 transformPoint(AnimationState s, vec3 p)
{
  return vec4(p, 1.f) * s.transform3x4;
}

void mulTransform(inout AnimationState s, mat4 m)
{
  setTransform4x4(s, m * getTransform4x4(s));
}


mat3x4 multiplyTransformMatrices(mat3x4 A, mat3x4 B)
{
  mat3x4 result;

  // Loop through rows of A
  [[unroll]] for(int i = 0; i < 3; i++)
  {
    // Loop through columns of B (since it's 3x4, we manually handle the last column)
    [[unroll]] for(int j = 0; j < 4; j++)
    {
      if(j < 3)
      {
        // Multiply the (i, j) element of the resulting matrix
        result[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j];
      }
      else
      {
        // Handle the last column (translation part)
        result[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j] + A[i][3];  // The translation component
      }
    }
  }

  return result;
}


void mulTransform3x4(inout AnimationState s, mat3x4 m)
{
  s.transform3x4 = multiplyTransformMatrices(m, s.transform3x4);
}

// From an index in the global object list of the scene, return the domino index in the dynamic object list
uint32_t globalIndexToDynamicIndex(uint32_t index, AnimationShaderData d)
{
  uint32_t staticObjectCount = d.totalObjectCount - d.dynamicObjectCount;

  if(index < staticObjectCount || index >= d.totalObjectCount)
  {
    return ~0u;
  }

  return index - staticObjectCount;
}

// From a domino index, return the corresponding object index in the global object list of the scene
uint32_t dynamicIndexToGlobalIndex(uint32_t index, AnimationShaderData d)
{
  uint32_t staticObjectCount = d.totalObjectCount - d.dynamicObjectCount;

  return index + staticObjectCount;
}


#else
inline void setTransform4x4(AnimationState& s, const glm::mat4& m)
{
  s.transform3x4 = mat3x4(transpose(m));
}

inline vec3 getPosition(const AnimationState& s)
{
  return vec3(s.transform3x4[0].w, s.transform3x4[1].w, s.transform3x4[2].w);
}
#endif


INSPECTABLE_STRUCT(PartitionState, {
  // Last timestamp when an object within the partition has been updated
  uint32_t lastModified;
  // Number of static objects in the partition
  uint32_t staticObjectCount;
  // Index of the first available slot in the partition's instance index list where
  // the instance update kernel can write the dynamic objects. This is also used as the
  // dynamic object count for the partition in the partition update kernel
  uint32_t dynamicWriteSlot;
  // If != 0, the instance index list of the partition has to be rewritten due to a domino being
  // move to/from the global partition
  uint32_t needInstanceIndicesRewrite;
});

struct PushConstant
{
  float metallic;
  float roughness;
  float intensity;
  int   maxDepth;

  AnimationShaderData animationShaderData;
};

// GLSL-compliant version of the VkAccelerationStructureInstance struct
// Note some members have been merged into a single members as
// GLSL does not support the bit-field syntax
struct AccelerationStructureInstance
{
  mat3x4   transform;
  uint32_t instanceID24Mask8;
  uint32_t instanceShaderBindingTableRecordOffset24Flags8;
  uint64_t accelerationStructureReference;
};


#endif  // HOST_DEVICE_H
