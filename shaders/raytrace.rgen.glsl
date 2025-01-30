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
#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference : require

#include "device_host.h"
#include "payload.h"
#include "dh_bindings.h"
#include "nvvkhl/shaders/random.h"
#include "nvvkhl/shaders/constants.h"

// clang-format off
layout(location = 0) rayPayloadEXT HitPayload payload;

layout(set = 0, binding = B_tlas) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = B_outImage, rgba32f) uniform image2D image;
layout(set = 0, binding = B_aoImage, rgba32f) uniform image2D imageAO;
layout(set = 0, binding = B_depthImage, r32f) uniform image2D imageDepth;
layout(set = 0, binding = B_frameInfo, scalar) uniform FrameInfo_ { FrameInfo frameInfo; };
// clang-format on

layout(push_constant) uniform RtxPushConstant_
{
  PushConstant pc;
};


layout(buffer_reference, scalar) readonly buffer AnimationStateBuffer
{
  AnimationState s[];
};

layout(buffer_reference, scalar) readonly buffer AnimationGlobalStateBuffer
{
  AnimationGlobalState s;
};

void main()
{
  uint32_t samplesPerAxis = 1;
  vec3     result         = vec3(0.f);
  vec3     aoResult       = vec3(0.f);
  uint32_t primaryHitId   = ~0u;

  for(uint32_t y = 0; y < samplesPerAxis; y++)
  {
    for(uint32_t x = 0; x < samplesPerAxis; x++)
    {
      payload          = initPayload(x + y * samplesPerAxis);
      const vec2 delta = vec2(float(x + 1) / float(samplesPerAxis + 1), float(y + 1) / float(samplesPerAxis + 1));
      const vec2 inUV  = (vec2(gl_LaunchIDEXT.xy) + delta) / vec2(gl_LaunchSizeEXT.xy);

      const vec2 d      = inUV * 2.0 - 1.0;
      const vec4 target = normalize(frameInfo.projInv * vec4(d.x, d.y, 1.0, 1.0));

      vec3 rayOrigin    = vec3(frameInfo.viewInv * vec4(0.0, 0.0, 0.0, 1.0));
      vec3 rayDirection = normalize(vec3(frameInfo.viewInv * vec4(target.xyz, 0.0)));

      const uint  rayFlags = gl_RayFlagsCullBackFacingTrianglesEXT;
      const float tMin     = 0.001;
      const float tMax     = INFINITE;

      traceRayEXT(topLevelAS,    // acceleration structure
                  rayFlags,      // rayFlags
                  0xFF,          // cullMask
                  0,             // sbtRecordOffset
                  0,             // sbtRecordStride
                  0,             // missIndex
                  rayOrigin,     // ray origin
                  tMin,          // ray min range
                  rayDirection,  // ray direction
                  tMax,          // ray max range
                  0              // payload (location = 0)
      );
      result += payload.color.xyz;
      aoResult += payload.primary.xyz;
      primaryHitId = payload.primaryHitId;
    }
  }
  result /= (samplesPerAxis * samplesPerAxis);
  aoResult /= (samplesPerAxis * samplesPerAxis);

  imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(result, payload.ao));

  float hitId = (primaryHitId & 0xFF) / 255.f;
  imageStore(imageAO, ivec2(gl_LaunchIDEXT.xy), vec4(aoResult, hitId));
  imageStore(imageDepth, ivec2(gl_LaunchIDEXT.xy), vec4(payload.primaryDepth, 0.f, 0.f, 0.f));
}
