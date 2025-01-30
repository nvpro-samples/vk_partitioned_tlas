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
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable

#include "device_host.h"
#include "dh_bindings.h"
#include "payload.h"
#include "nvvkhl/shaders/constants.h"
#include "nvvkhl/shaders/dh_sky.h"
#include "nvvkhl/shaders/func.h"
#include "nvvkhl/shaders/ggx.h"
#include "nvvkhl/shaders/random.h"

hitAttributeEXT vec2 attribs;

// clang-format off
layout(location = 0) rayPayloadInEXT HitPayload payload;

layout(set = 0, binding = B_tlas ) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = B_frameInfo, scalar) uniform FrameInfo_ { FrameInfo frameInfo; };
layout(set = 0, binding = B_skyParam,  scalar) uniform SkyInfo_ { SimpleSkyParameters skyInfo; };
layout(set = 0, binding = B_materials, scalar) buffer Materials_ { vec4 m[]; } materials;
layout(set = 0, binding = B_instances, scalar) buffer InstanceInfo_ { InstanceInfo i[]; } instanceInfo;
layout(set = 0, binding = B_vertex, scalar) buffer Vertex_ { Vertex v[]; } vertices[];
layout(set = 0, binding = B_index, scalar) buffer Index_ { uvec3 i[]; } indices[];

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

vec3 getPartitionPosition(uint32_t partitionId)
{
  uint32_t x = partitionId % pc.animationShaderData.partitionsPerAxis;
  uint32_t y = partitionId / pc.animationShaderData.partitionsPerAxis;

  vec3 bbSize = pc.animationShaderData.objectBboxMax - pc.animationShaderData.objectBboxMin;

  float dx = (0.5f + float(x)) * (bbSize.x / float(pc.animationShaderData.partitionsPerAxis));
  float dy = (0.5f + float(y)) * (bbSize.z / float(pc.animationShaderData.partitionsPerAxis));

  return pc.animationShaderData.objectBboxMin + vec3(dx, 0, dy);
}
//-----------------------------------------------------------------------
// Hit state information
struct HitState
{
  vec3 pos;
  vec3 nrm;
  vec3 geonrm;
  vec2 texCoord;
};

//-----------------------------------------------------------------------
// Return hit position and normal in world space
HitState getHitState(int meshID, vec3 barycentrics)
{
  HitState hit;

  // Getting the 3 indices of the triangle (local)
  uvec3 triangleIndex = indices[meshID].i[gl_PrimitiveID];

  // All vertex attributes of the triangle.
  Vertex v0 = vertices[meshID].v[triangleIndex.x];
  Vertex v1 = vertices[meshID].v[triangleIndex.y];
  Vertex v2 = vertices[meshID].v[triangleIndex.z];

  // Position
  const vec3 pos0     = v0.position.xyz;
  const vec3 pos1     = v1.position.xyz;
  const vec3 pos2     = v2.position.xyz;
  const vec3 position = pos0 * barycentrics.x + pos1 * barycentrics.y + pos2 * barycentrics.z;
  hit.pos             = vec3(gl_ObjectToWorldEXT * vec4(position, 1.0));

  // Normal
  const vec3 nrm0   = v0.normal.xyz;
  const vec3 nrm1   = v1.normal.xyz;
  const vec3 nrm2   = v2.normal.xyz;
  const vec3 normal = normalize(nrm0 * barycentrics.x + nrm1 * barycentrics.y + nrm2 * barycentrics.z);

  vec3       worldNormal    = normalize(vec3(normal * gl_WorldToObjectEXT));
  const vec3 geoNormal      = normalize(cross(pos1 - pos0, pos2 - pos0));
  vec3       worldGeoNormal = normalize(vec3(geoNormal * gl_WorldToObjectEXT));
  hit.geonrm                = worldGeoNormal;
  hit.nrm                   = worldNormal;

  hit.texCoord = v0.texCoord * barycentrics.x + v1.texCoord * barycentrics.y + v2.texCoord * barycentrics.z;

  return hit;
}

//-----------------------------------------------------------------------
// Return TRUE if there is no occluder, meaning that the light is visible from P toward L
bool shadowRay(vec3 P, vec3 L, float tmax)
{
  const uint rayFlags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT | gl_RayFlagsCullBackFacingTrianglesEXT;
  HitPayload savedP = payload;
  traceRayEXT(topLevelAS, rayFlags, 0xFF, 0, 0, 0, P, 0.001, L, tmax, 0);
  bool visible = (payload.depth == MISS_DEPTH);
  payload      = savedP;
  return visible;
}

#ifndef GGX_MIN_ALPHA_ROUGHNESS
#define GGX_MIN_ALPHA_ROUGHNESS 1e-03F
#endif

float ggxSmithVisibility(float NdotL, float NdotV, float alphaRoughness)
{
  alphaRoughness = max(alphaRoughness, GGX_MIN_ALPHA_ROUGHNESS);
  float alphaRoughnessSq = alphaRoughness * alphaRoughness;

  float ggxV = NdotL * sqrt(NdotV * NdotV * (1.0F - alphaRoughnessSq) + alphaRoughnessSq);
  float ggxL = NdotV * sqrt(NdotL * NdotL * (1.0F - alphaRoughnessSq) + alphaRoughnessSq);

  return 0.5F / (ggxV + ggxL);
}
float ggxDistribution(float NdotH, float alphaRoughness)  // alphaRoughness    = roughness * roughness;
{
  if (NdotH < 0.0f)
  {
    return 0.0f;
  }
  alphaRoughness = max(alphaRoughness, GGX_MIN_ALPHA_ROUGHNESS);
  float alphaSqr = alphaRoughness * alphaRoughness;

  float NdotHSqr = NdotH * NdotH;
  float denom = NdotHSqr * (alphaSqr - 1.0F) + 1.0F;

  return alphaSqr / (M_PI * denom * denom);
}

vec3 ggxEvaluate(vec3 V, vec3 N, vec3 L, vec3 albedo, float metallic, float roughness)
{
  // Specular reflection
  vec3  H              = normalize(L + V);
  float alphaRoughness = roughness;
  float NdotL          = clampedDot(N, L);
  float NdotV          = clampedDot(N, V);
  float NdotH          = clampedDot(N, H);
  float VdotH          = clampedDot(V, H);

  vec3  c_min_reflectance = vec3(0.04);
  vec3  f0                = mix(c_min_reflectance, albedo, metallic);
  vec3  f90               = vec3(1.0);
  vec3  f                 = schlickFresnel(f0, f90, VdotH);
  float vis               = ggxSmithVisibility(NdotL, NdotV, alphaRoughness);  // Vis = G / (4 * NdotL * NdotV)
  float d                 = ggxDistribution(NdotH, alphaRoughness);

  vec3 bsdf_glossy = f * (vis * d) * NdotL;  // GGX-Smith
  vec3 bsdf_diffuse =
      albedo * schlickFresnel(1.0F - c_min_reflectance, vec3(0.0F), VdotH) * (1.0F - metallic) * (M_1_PI * NdotL);  // Lambertian


  return bsdf_glossy + bsdf_diffuse;
}


vec3 uintToColor(uint cellId)
{
  cellId = pcg(cellId);
  uint r = cellId & 0xFF;
  uint g = (cellId >> 8) & 0xFF;
  uint b = (cellId >> 16) & 0xFF;
  return vec3(float(r) / 255.0, float(g) / 255.0, float(b) / 255.0);
}


float ambientOcclusion(vec3 wPos, vec3 wNormal, uint32_t sampleCount, uint32_t sampleOffset)
{
  vec3     normal = wNormal;
  vec3     tangent, bitangent;
  orthonormalBasis(normal, tangent, bitangent);

  // FIXME: breaks low-discrepancy sequence
  uint32_t seed = (sampleOffset * gl_LaunchSizeEXT.y + gl_LaunchIDEXT.y) * gl_LaunchSizeEXT.x + gl_LaunchIDEXT.x;

  uint32_t occlusion = 0u;

  for(uint32_t i = 0; i < sampleCount; i++)
  {
    float r1 = rand(seed);
    float r2 = rand(seed);
    vec3 wDirection = cosineSampleHemisphere(r1, r2);
    wDirection      = wDirection.x * tangent + wDirection.y * bitangent + wDirection.z * normal;
    if(!shadowRay(wPos, wDirection, 5.f))
    {
      occlusion++;
    }
  }

  return max(0.2f, float(sampleCount - occlusion) / float(sampleCount));
}

vec3 desaturate(vec3 color, float desaturation)
{
  // Convert the input color to grayscale using luminosity method
  float gray = dot(color, vec3(0.299, 0.587, 0.114));

  // Interpolate between the original color and the grayscale color based on the desaturation coefficient
  return mix(color, vec3(gray), desaturation);
}


void addColor(vec3 col)
{
  if(payload.depth == 0)
  {
    payload.primary += col;
  }
  else
  {
    payload.color += col;
  }
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
void main()
{
  // We hit our max depth
  if(payload.depth >= pc.maxDepth - 1)
  {
    return;
  }

  vec3 P = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;
  vec3 D = normalize(gl_WorldRayDirectionEXT);
  vec3 V = -D;

  // Vector to the light
  vec3 L = normalize(skyInfo.directionToLight);
  // Compute hard shadows
  bool visible = shadowRay(P, L, 1e34f);


  uint32_t instanceID = gl_InstanceID & 0x00FFFFFF;

  // Retrieve the Instance buffer information
  InstanceInfo iInfo = instanceInfo.i[instanceID];

  // Barycentric coordinate on the triangle
  const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

  int32_t meshId = iInfo.meshID;


  HitState hit = getHitState(meshId, barycentrics);

  if(payload.depth == 0)
  {
    // For primary hits, compute a per-pixel hash that will be used for edge detection
    payload.primaryHitId = xxhash32(floatBitsToUint(hit.geonrm));
  }


  vec3           albedo      = materials.m[iInfo.materialID].xyz;
  float          alpha       = materials.m[iInfo.materialID].w;
  bool           isDominoHit = false;
  bool           colorRed    = false;
  AnimationState animState;

  // If a dynamic object is hit, check whether the location corresponds to the mouse coordinate and a topple is requested.
  // If yes, update the state of the domino and request a topple.
  uint32_t dynamicIndex = globalIndexToDynamicIndex(instanceID, pc.animationShaderData);
  if(dynamicIndex != ~0u)
  {
    isDominoHit = true;
    animState = AnimationStateBuffer(pc.animationShaderData.state[(pc.animationShaderData.currentStateIndex) % 2]).s[dynamicIndex];
    colorRed = animState.firstContact != ~0u && (pc.animationShaderData.frameIndex - animState.firstContact) < 10;
    if(gl_LaunchIDEXT.xy == pc.animationShaderData.mouseCoord)
    {
      if(pc.animationShaderData.toppleDomino != 0)
      {
        addFlag(AnimationStateBuffer(pc.animationShaderData.state[0]).s[dynamicIndex].stateID, STATE_FORCE_TOPPLE);
        addFlag(AnimationStateBuffer(pc.animationShaderData.state[1]).s[dynamicIndex].stateID, STATE_FORCE_TOPPLE);

        AnimationGlobalStateBuffer(pc.animationShaderData.globalState).s.toppleRequest = dynamicIndex;

        colorRed = true;
      }
    }
  }

  if(colorRed)
  {
    albedo = vec3(1.f, 0.f, 0.f);
  }

  float metallic = pc.metallic;

  bool isActivePartition = false;

  // For PTLAS, color the dominoes and the static objects based on the partition ID
  if(pc.animationShaderData.ptlasActive != 0)
  {
    // Domino shading
    if(globalIndexToDynamicIndex(instanceID, pc.animationShaderData) != ~0u)
    {
      if(!colorRed)
      {
        albedo = uintToColor(animState.partitionID);
      }
#if 0
      // If the dynamic update behavior depends on the viewing distance, color half of the domino with the color of the global partition
      if(payload.depth == 0 && pc.animationShaderData.dynamicUpdateMode == PTLAS_DYNAMIC_UPDATE_OR_MOVE_TO_GLOBAL)
      {
        vec3  partitionPos = getPartitionPosition(animState.partitionID);
        vec3  toCam        = partitionPos - pc.animationShaderData.eyePosition;
        float sqDist       = dot(toCam, toCam);
        if(sqDist >= pc.animationShaderData.dynamicDistanceThreshold * pc.animationShaderData.dynamicDistanceThreshold)
        {
          if(hit.texCoord.x < 0.5f)
          {
            albedo = desaturate(uintToColor(pc.animationShaderData.globalPartitionIndex) * 0.7f, 0.8f);
          }
        }
      }
#endif
    }
    // Static object shading
    else
    {
      // Compute the partition ID based on the object's position
      vec3 bboxSize = pc.animationShaderData.objectBboxMax - pc.animationShaderData.objectBboxMin;


      vec3 centerPos = vec3(gl_ObjectToWorldEXT[3]);


      uint32_t partitionX = uint32_t(((centerPos.x - pc.animationShaderData.objectBboxMin.x) / bboxSize.x)
                                     * (pc.animationShaderData.partitionsPerAxis));
      uint32_t partitionY = uint32_t(((centerPos.z - pc.animationShaderData.objectBboxMin.z) / bboxSize.z)
                                     * (pc.animationShaderData.partitionsPerAxis));

      uint32_t       partitionID = partitionX + partitionY * pc.animationShaderData.partitionsPerAxis;
      PartitionState p           = PartitionStateBuffer(pc.animationShaderData.partitionState).s[partitionID];

      isActivePartition = (p.lastModified != ~0u && (pc.animationShaderData.frameIndex - p.lastModified) < 10);

      // If the partition is active, ie. it is currently being updated, color the object with the partition color
      if(isActivePartition)
      {
        albedo = uintToColor(partitionID) * 0.7f;
        if(payload.depth == 0)
        {
          payload.primaryHitId += partitionID;
        }
      }
      else
      {
        // If the partition is not active, desaturate the object color
        albedo = mix(albedo, desaturate(uintToColor(partitionID) * 0.7f, 0.8f), 0.4f);
      }


      // Desaturate the static objects to increase the visibility of the dominoes
      albedo = mix(albedo, desaturate(uintToColor(partitionID) * 0.7f, 0.8f), 0.4f);
    }
  }


  // Color at hit point
  vec3 color = ggxEvaluate(V, hit.nrm, L, albedo, metallic, pc.roughness);


  float ao = 1.f;

  // Compute ambient occlusion on primary hits
  if(payload.depth < 1)
  {
    const uint32_t sampleCount = 2;
    ao                   = min(0.99f, ambientOcclusion(P, hit.nrm, sampleCount, sampleCount * payload.sampleIndex));
    payload.ao           = ao;
    payload.primaryDepth = gl_HitTEXT;
  }

  // Account for the hard shadowing
  if(!visible)
  {
    if(isActivePartition)
    {
      color *= 0.7f;
    }
    else
    {
      color *= 0.3F;
    }
  }
  addColor(color * payload.weight * pc.intensity);

  HitPayload backup = payload;
  vec3       col    = payload.color;

  {
    payload.color = vec3(0);
    // Reflection
    vec3 refl_dir = reflect(D, hit.nrm);

    payload.depth += 1;
    payload.weight *= metallic;  // more or less reflective

    traceRayEXT(topLevelAS, gl_RayFlagsCullBackFacingTrianglesEXT, 0xFF, 0, 0, 0, P, 0.0001, refl_dir, 100.0, 0);
    col += payload.color;
  }
  // For every 10th domino, or if the object is transparent, compute a very simplistic refraction
  if((instanceID % 10 == 0 && globalIndexToDynamicIndex(instanceID, pc.animationShaderData) != ~0u) || alpha < 1.f)
  {
    payload       = backup;
    payload.color = vec3(0);

    vec3 refractNormal = hit.nrm;
    refractNormal.x    = refractNormal.x
                      + 0.01f * sin(100.f * P.x + pc.animationShaderData.frameIndex / 50.f)
                            * cos(320.f * P.z + pc.animationShaderData.frameIndex / 50.f);
    refractNormal.y = refractNormal.y
                      + 0.01f * cos(237.f * P.y + pc.animationShaderData.frameIndex / 50.f)
                            * sin(20.f * P.x + pc.animationShaderData.frameIndex / 50.f);
    refractNormal.z = refractNormal.z
                      + 0.01f * sin(75.f * P.z + pc.animationShaderData.frameIndex / 50.f)
                            * cos(52.f * P.y + pc.animationShaderData.frameIndex / 50.f);

    refractNormal = normalize(refractNormal);

    vec3 reflectedDirection = refract(D, refractNormal, 1.2f);

    payload.depth += 1;


    if(alpha < 1.f)
    {
      payload.weight *= vec3(alpha);  // more or less reflective
    }
    else
    {
      payload.weight *= pc.metallic;  // more or less reflective
    }
    // Trace the reflected ray
    traceRayEXT(topLevelAS, gl_RayFlagsCullBackFacingTrianglesEXT, 0xFF, 0, 0, 0, P, 0.001, reflectedDirection, 100.0, 0);

    col = col * 0.2f + payload.color;
  }

  payload.color = col;
}
