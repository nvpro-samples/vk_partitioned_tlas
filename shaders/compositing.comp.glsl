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
  Final compositing shader that combines the ray-traced image with the ambient occlusion and applies cell shading and edge detection.

*/

#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_atomic_float : enable
#extension GL_EXT_control_flow_attributes2 : enable

#include "device_host.h"
#include "dh_bindings.h"

layout(push_constant) uniform CompositingPushConstant_
{
  CompositingShaderData constants;
};


layout(set = 0, binding = B_outImage, rgba32f) uniform image2D image;
layout(set = 0, binding = B_aoImage, rgba32f) uniform image2D imageAO;
layout(set = 0, binding = B_depthImage, rgba32f) uniform image2D imageDepth;

layout(local_size_x = COMPOSITING_SHADER_BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;

int32_t increment(int32_t existing)
{
  if(abs(existing) < 2)
  {
    return existing + 1;
  }
  if(abs(existing) < 5)
  {
    return existing + 2;
  }
  return existing + 4;
}


float noise(float x)
{
  return fract(sin(x) * 43758.5453);
}


vec3 medianFilter(ivec2 coord)
{
  // Offset values for neighboring pixels
  ivec2 offsets[9] = {ivec2(-1, -1), ivec2(0, -1), ivec2(1, -1), ivec2(-1, 0), ivec2(0, 0),
                      ivec2(1, 0),   ivec2(-1, 1), ivec2(0, 1),  ivec2(1, 1)};

  // Array to store neighboring pixel values
  vec3 neighborValues[9];

  // Sample neighboring pixels and store in array
  for(int i = 0; i < 9; ++i)
  {
    ivec2 neighborCoord = coord + offsets[i];
    // Handle out-of-bounds access (clamp or wrap as needed)
    neighborCoord     = clamp(neighborCoord, ivec2(0, 0), imageSize(imageAO));
    neighborValues[i] = vec3(imageLoad(imageAO, neighborCoord).rgb);
  }

  // Sort the array of neighboring pixel values
  for(int i = 0; i < 8; ++i)
  {
    for(int j = i + 1; j < 9; ++j)
    {
      if(neighborValues[i].r > neighborValues[j].r)
      {
        vec3 temp         = neighborValues[i];
        neighborValues[i] = neighborValues[j];
        neighborValues[j] = temp;
      }
    }
  }

  // Return the median value (element at index 4)
  return neighborValues[4];
}


float grayscale(vec3 c)
{
  return c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722;
}


void main()
{
  uint32_t index = gl_GlobalInvocationID.x;

  uint32_t resX = imageSize(image).x;
  uint32_t resY = imageSize(image).y;

  uint32_t pixelCount = resX * imageSize(image).y;

  if(index >= pixelCount)
  {
    return;
  }

  ivec2 coord = ivec2(index % resX, index / resX);

  if(constants.passId == 0)  // AO filtering
  {
    vec4 imagePixel = imageLoad(image, coord);
    int  halfWindow = 20;

    vec4  centerAoPixel = imageLoad(imageAO, coord);
    float centerId      = centerAoPixel.w;
    if(centerId == 1.f)
    {
      return;
    }
    float centerAo = imagePixel.w;

    float aoResult = centerAo;
    float aoWeight = 1.f;


    for(int32_t y = -halfWindow; y <= halfWindow; y = increment(y))
    {
      ivec2 localCoord;
      localCoord.y = coord.y + y;
      if(localCoord.y < 0 || localCoord.y >= resY)
      {
        continue;
      }
      for(int32_t x = -halfWindow; x <= halfWindow; x = increment(x))
      {
        localCoord.x = coord.x + x;
        if(localCoord.x < 0 || localCoord.x >= resX || (x == 0 && y == 0))
        {
          continue;
        }
        float localAoPixel = imageLoad(image, localCoord).w;
        float localId      = imageLoad(imageAO, localCoord).w;
        if(localId == centerId)
        {
          float maxDist     = sqrt(2.f * halfWindow * halfWindow);
          float distance    = sqrt(float(x * x + y * y));
          float localWeight = (maxDist - distance) / maxDist;
          aoResult += localAoPixel * localWeight;


          aoWeight += localWeight;
        }
      }
    }
    aoResult /= float(aoWeight);

    vec3 finalColor = imagePixel.xyz + aoResult * centerAoPixel.xyz;

    float centerDepth = imageLoad(imageDepth, coord).x;
    if(centerDepth > 1000.f)
    {
      vec3 haze  = vec3(0.7);
      finalColor = mix(finalColor, haze, min(0.8f, (centerDepth - 1000.f) / 1000.f));
    }

    imageStore(image, coord, vec4(finalColor, centerAo));

    return;
  }


  if(constants.passId == 1)  // Edge detection
  {
    float centerDepth = imageLoad(imageDepth, coord).x;

    vec4  centerAoPixel = imageLoad(imageAO, coord);
    float centerId      = centerAoPixel.w;
    vec4  centerPixel   = imageLoad(image, coord);

    bool isEdge = false;

    int32_t halfWindow = 1;

    uint32_t edgesFound = 0;
    uint32_t testCount  = 0;
    for(int32_t y = -halfWindow; y <= halfWindow; y++)
    {
      ivec2 localCoord;
      localCoord.y = coord.y + y;
      if(localCoord.y < 0 || localCoord.y >= resY)
      {
        continue;
      }
      for(int32_t x = -halfWindow; x <= halfWindow; x++)
      {
        localCoord.x = coord.x + x;
        if(localCoord.x < 0 || localCoord.x >= resX || (x == 0 && y == 0))
        {
          continue;
        }
        testCount++;
        vec4  localAoPixel = imageLoad(imageAO, localCoord);
        float localId      = localAoPixel.w;
        if(localId < centerId || (centerDepth < 10.f && localId != centerId))
        {
          isEdge = true;
          edgesFound++;
          break;
        }
      }
      if(isEdge)
      {
        break;
      }
    }

    float edgeValue = edgesFound > 0 ? 0.f : 1.f;

    imageStore(image, coord, vec4(centerPixel.xyz, edgeValue));
    imageStore(imageAO, coord, vec4(centerPixel.xyz, centerId));

    return;
  }


  if(constants.passId == 2)  // Cell shading
  {

    float centerDepth   = imageLoad(imageDepth, coord).x;
    vec4  centerAoPixel = imageLoad(imageAO, coord);
    float centerId      = centerAoPixel.w;


    const float numGrays = 10;
    vec3        median   = medianFilter(coord);
    float       gray     = grayscale(median);

    vec3  color     = median;
    float grayLevel = float(int(gray * numGrays)) / ((numGrays - 1.) * grayscale(color));
    color *= grayLevel;

    vec4 centerPixel = imageLoad(image, coord);


    color = mix(color, centerPixel.xyz, clamp((centerDepth - 100.f) / 100.f, 0.5f, 1.f));


    imageStore(image, coord, vec4(color, centerPixel.w));
    return;
  }


  if(constants.passId == 3)  // Add edges
  {

    const vec4  edgeColor       = vec4(0.2, 0.2, 0.15, 1.0);
    const vec4  backgroundColor = vec4(1, 0.95, 0.85, 1);
    const float noiseAmount     = 0.00;
    const float errorPeriod     = 30.0;
    float       errorRange      = 0.001;

    float centerDepth = imageLoad(imageDepth, coord).x;
    if(centerDepth > 5.f)
    {
      errorRange *= 1.f - min(1.f, (centerDepth - 5.f) / 10.f);
    }

    vec2 uv = vec2(coord) / vec2(resX, resY);

    float center = imageLoad(image, coord).w;

    if(center > 0.1f && center < 0.6f)
    {
      errorRange = 0.f;
    }

    float noise = uv.x * noiseAmount;
    vec2  uvs[3];
    uvs[0] = uv + vec2(errorRange * sin(errorPeriod * uv.y + 0.0) + noise, errorRange * sin(errorPeriod * uv.x + 0.0) + noise);
    uvs[1] = uv + vec2(errorRange * sin(errorPeriod * uv.y + 1.047) + noise, errorRange * sin(errorPeriod * uv.x + 3.142) + noise);
    uvs[2] = uv + vec2(errorRange * sin(errorPeriod * uv.y + 2.094) + noise, errorRange * sin(errorPeriod * uv.x + 1.571) + noise);

    float e0 = imageLoad(image, ivec2(uvs[0] * vec2(resX, resY))).w;
    float e1 = imageLoad(image, ivec2(uvs[1] * vec2(resX, resY))).w;
    float e2 = imageLoad(image, ivec2(uvs[2] * vec2(resX, resY))).w;

    float edge = e0 * e1 * e2;


    float depthWeight = clamp(centerDepth / 100.f, 0.f, 1.f);
    edge              = max(edge, depthWeight);
    if(edge < 1.f)
    {
      edge *= depthWeight;
    }
    vec4 centerPixel = imageLoad(image, coord);
    imageStore(image, coord, vec4(centerPixel.xyz * edge, centerPixel.w));
    return;
  }
}