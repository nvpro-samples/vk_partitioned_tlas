

/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#ifndef FUNCTIONS_HLSLI
#define FUNCTIONS_HLSLI 1

float square(float x)
{
  return x * x;
}

float clampedDot(float3 x, float3 y)
{
  return clamp(dot(x, y), 0.0F, 1.0F);
}

//-----------------------------------------------------------------------
// Building an Orthonormal Basis, Revisited
// by Tom Duff, James Burgess, Per Christensen, Christophe Hery, Andrew Kensler, Max Liani, Ryusuke Villemin
// https://graphics.pixar.com/library/OrthonormalB/
//-----------------------------------------------------------------------
void orthonormalBasis(in float3 normal, out float3 tangent, out float3 bitangent)
{
  float sgn = normal.z > 0.0F ? 1.0F : -1.0F;
  float a = -1.0F / (sgn + normal.z);
  float b = normal.x * normal.y * a;

  tangent = float3(1.0F + sgn * normal.x * normal.x * a, sgn * b, -sgn * normal.x);
  bitangent = float3(b, sgn + normal.y * normal.y * a, -normal.y);
}


//-----------------------------------------------------------------------
// https://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.4.pdf
// 16.6.1 COSINE-WEIGHTED HEMISPHERE ORIENTED TO THE Z-AXIS
//-----------------------------------------------------------------------
float3 cosineSampleHemisphere(float r1, float r2)
{
  const float M_TWO_PI = 6.2831853071795F;
  float r = sqrt(r1);
  float phi = M_TWO_PI * r2;
  float3 dir;
  dir.x = r * cos(phi);
  dir.y = r * sin(phi);
  dir.z = sqrt(max(0.0, 1.0 - dir.x * dir.x - dir.y * dir.y));
  return dir;
}

//-------------------------------------------------------------------------------------------------
// Avoiding self intersections
//-----------------------------------------------------------------------
float3 offsetRay(in float3 p, in float3 n)
{
  // Smallest epsilon that can be added without losing precision is 1.19209e-07, but we play safe
  const float epsilon = 1.0F / 65536.0F; // Safe epsilon

  float magnitude = length(p);
  float offset = epsilon * magnitude;
  // multiply the direction vector by the smallest offset
  float3 offsetVector = n * offset;
  // add the offset vector to the starting point
  float3 offsetPoint = p + offsetVector;

  return offsetPoint;
}

// Hacking the shadow terminator
// https://jo.dreggn.org/home/2021_terminator.pdf
// p : point of intersection
// p[a..c]: position of the triangle
// n[a..c]: normal of the triangle
// bary: barycentric coordinate of the hit position
// return the offset position
float3 pointOffset(float3 p, float3 pa, float3 pb, float3 pc, float3 na, float3 nb, float3 nc, float3 bary)
{
  float3 tmpu = p - pa;
  float3 tmpv = p - pb;
  float3 tmpw = p - pc;

  float dotu = min(0.0F, dot(tmpu, na));
  float dotv = min(0.0F, dot(tmpv, nb));
  float dotw = min(0.0F, dot(tmpw, nc));

  tmpu -= dotu * na;
  tmpv -= dotv * nb;
  tmpw -= dotw * nc;

  float3 pP = p + tmpu * bary.x + tmpv * bary.y + tmpw * bary.z;

  return pP;
}

// Convert an integer to a color. This can help visualy debugging values.
float3 IntegerToColor(uint val)
{
  const float3 freq = float3(1.33333f, 2.33333f, 3.33333f);
  return float3(sin(freq * val) * .5 + .5);
}

// Return the luminance of a color
float luminance(in float3 color)
{
  return color.x * 0.2126F + color.y * 0.7152F + color.z * 0.0722F;
}


// Utility for temperature and landscapeColor:
// Smoothly transforms the range [low, high] to [0, 1], with 0 derivative at
// low, high, and (low + high) * 0.5.
float fade(float low, float high, float value)
{
  float mid = (low + high) * 0.5;
  float range = (high - low) * 0.5;
  float x = 1.0 - clamp(abs(mid - value) / range, 0.0, 1.0);
  return smoothstep(0.0, 1.0, x);
}

// Return a cold-hot color based on intensity [0-1]
float3 temperature(float intensity)
{
  const float3 blue   = float3(0.0, 0.0, 1.0);
  const float3 cyan   = float3(0.0, 1.0, 1.0);
  const float3 green  = float3(0.0, 1.0, 0.0);
  const float3 yellow = float3(1.0, 1.0, 0.0);
  const float3 red    = float3(1.0, 0.0, 0.0);

  float3 color = (fade(-0.25, 0.25, intensity) * blue    //
                  + fade(0.0, 0.5, intensity) * cyan     //
                  + fade(0.25, 0.75, intensity) * green  //
                  + fade(0.5, 1.0, intensity) * yellow   //
                  + smoothstep(0.75, 1.0, intensity) * red);
  return color;
}

// Return a landscape color based on height [0-1]
float3 landscapeColor(float height)
{
  const float3 water = float3(0.0, 0.0, 0.5);
  const float3 sand = float3(0.8, 0.7, 0.4);
  const float3 green = float3(0.1, 0.4, 0.1);
  const float3 rock = float3(0.4, 0.4, 0.4);
  const float3 snow = float3(1.0, 1.0, 1.0);


  float3 color = (fade(-0.25, 0.25, height) * water //
                + fade(0.0, 0.5, height) * sand //
                + fade(0.25, 0.75, height) * green //
                + fade(0.5, 1.0, height) * rock //
                + smoothstep(0.75, 1.0, height) * snow);
  return color;
}

// Performs shading calculations for a 3D surface point, taking into account light direction, view direction,
// surface color, and surface normal. It combines diffuse and specular components to determine the light
// intensity, multiplies it with the surface color, adds an ambient term (sky), and returns the final shaded color.
float3 simpleShading(float3 viewDir, float3 lightDir, float3 normal, float3 color = float3(1, 1, 1), float expo = 16.0)
{
  // Diffuse + Specular
  float3 reflDir = normalize(-reflect(lightDir, normal));
  float lt = saturate(dot(normal, lightDir)) + pow(max(0, dot(reflDir, viewDir)), expo);
  color *= lt;

  // Slight ambient term (sky effect)
  float3 skyUpDir = float3(0, 1, 0);
  float3 groundColor = float3(0.1, 0.1, 0.4);
  float3 skyColor = float3(0.8, 0.6, 0.2);
  color += lerp(skyColor, groundColor, dot(normal, skyUpDir.xyz) * 0.5 + 0.5) * 0.2;

  return color;
}

float4 makeFastTangent(float3 nrm)
{
  const float sgn = nrm.z > 0.0F ? 1.0F : -1.0F;
  const float a = -1.0F / (sgn + nrm.z);
  const float b = nrm.x * nrm.y * a;
  return float4(1.0f + sgn * nrm.x * nrm.x * a, sgn * b, -sgn * nrm.x, sgn);
}

#ifndef __SLANG__

float4 unpackUnorm4x8(uint packedValue) 
{
    float4 unpackedValue;

    // Unpack each component from the packed value
    unpackedValue.r = (packedValue & 0xFF) / 255.0f;
    unpackedValue.g = ((packedValue >> 8) & 0xFF) / 255.0f;
    unpackedValue.b = ((packedValue >> 16) & 0xFF) / 255.0f;
    unpackedValue.a = ((packedValue >> 24) & 0xFF) / 255.0f;

    return unpackedValue;
}
#endif

#if __HLSL_VERSION > 1

// This helper is used to load data from a buffer
// The startAddress is the address of the buffer
// The offset is the starting position in the buffer
//
// The loadValue function will load the data from the buffer
// NOTE: loadValue will increment the offset, therefore all the loadValue should be called in the same order as the data is stored in the buffer
struct LoaderHelper
{
  void init(uint64_t startAddress, uint64_t offset=0) { m_address = startAddress; m_offset = offset;}

  template<typename T>
  void loadValue(inout T value)
  {
    value = vk::RawBufferLoad<T>(m_address + m_offset);
    m_offset += sizeof(T);
  }

  uint64_t m_address;
  uint64_t m_offset;
};

#endif

#endif