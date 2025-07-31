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
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable

#include "shaderio.h"
#include "payload.h"

#include "nvshaders/slang_types.h"
#include "nvshaders/constants.h.slang"
#include "nvshaders/sky_io.h.slang"
#include "nvshaders/functions.h.slang"
#include "nvshaders/pbr_ggx_microfacet.h.slang"
#include "nvshaders/random.h.slang"
#include "nvshaders/sky_functions.h.slang"


// clang-format off


layout(set = 0, binding = B_frameInfo, scalar) uniform FrameInfo_ { FrameInfo frameInfo; };
layout(set = 0, binding = B_skyParam, scalar) uniform SkyInfo_ { SkySimpleParameters skyInfo; };
layout(set = 0, binding = B_materials, scalar) buffer Materials_ { vec4 m[]; } materials;
layout(set = 0, binding = B_instances, scalar) buffer InstanceInfo_ { InstanceInfo i[]; } instanceInfo;
layout(set = 0, binding = B_vertex, scalar) buffer Vertex_ { Vertex v[]; } vertices[];
layout(set = 0, binding = B_index, scalar) buffer Index_ { uvec3 i[]; } indices[];
layout(set = 0, binding = B_outImage, rgba32f) uniform image2D image;
layout(set = 0, binding = B_aoImage, rgba32f) uniform image2D imageAO;
layout(set = 0, binding = B_depthImage, r32f) uniform image2D imageDepth;

layout(push_constant) uniform RtxPushConstant_ { PushConstant pc; };

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

// clang-format on
