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

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_atomic_float : enable
#extension GL_EXT_control_flow_attributes2 : enable

#include "device_host.h"

layout(push_constant) uniform AnimationPushConstant_
{
  AnimationShaderData constants;
};

layout(buffer_reference, scalar) readonly buffer AccelerationStructureInstanceBuffer
{
  AccelerationStructureInstance i[];
};

layout(buffer_reference, scalar) readonly buffer AnimationStateBuffer
{
  AnimationState s[];
};

layout(buffer_reference, scalar) readonly buffer AnimationGlobalStateBuffer
{
  AnimationGlobalState s;
};
layout(buffer_reference, scalar) readonly buffer PartitionStateBuffer
{
  PartitionState s[];
};

layout(buffer_reference, scalar) readonly buffer PartitionedAccelerationStructureWriteInstanceDataNVBuffer
{
  PartitionedAccelerationStructureWriteInstanceDataNV s[];
};


layout(buffer_reference, scalar) readonly buffer PartitionedAccelerationStructureWritePartitionDataNVBuffer
{
  PartitionedAccelerationStructureWritePartitionDataNV s[];
};

layout(buffer_reference, scalar) readonly buffer BuildPartitionedAccelerationStructureIndirectCommandNVBuffer
{
  BuildPartitionedAccelerationStructureIndirectCommandNV s[];
};

layout(buffer_reference, scalar) readonly buffer U32Buffer
{
  uint32_t s[];
};


layout(local_size_x = ANIMATION_SHADER_BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;