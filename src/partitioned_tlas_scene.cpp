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

#include "partitioned_tlas.hpp"
#include "nvh/parallel_work.hpp"
#include <random>

// Compute a quaternion to orient a domino along the derivative of the curve
glm::quat computeQuaternion(const glm::vec2& derivative)
{
  if(derivative.x == 0.f && derivative.y == 0.f)
  {
    return glm::quat(1, 0, 0, 0);
  }
  glm::vec2 v = glm::normalize(derivative);

  float angle = acosf(v.y);
  if(v.x < 0.f)
    angle = -angle;
  glm::vec3 up(0.0f, 1.0f, 0.0f);
  return glm::angleAxis(angle + glm::pi<float>() / 2.f, up);
}


// Compute the number of dominoes within a curve segment
uint32_t computeSegmentRequirements(uint32_t index, const std::vector<glm::vec3>& cp, float spacing)
{
  // indices of the relevant control points
  int i0 = glm::clamp<int>(index - 1, 0, int(cp.size() - 1));
  int i1 = glm::clamp<int>(index, 0, int(cp.size() - 1));
  int i2 = glm::clamp<int>(index + 1, 0, int(cp.size() - 1));
  int i3 = glm::clamp<int>(index + 2, 0, int(cp.size() - 1));

  glm::vec3 from = cp[i1];
  glm::vec3 to   = cp[i2];

  float len = glm::length(from - to);

  uint32_t samples = uint32_t(ceil(len / spacing));

  return samples;
}

// Generate dominoes along a segment by sampling it uniformly. The locations of the dominos are added in res
void sampleSegment(uint32_t index, const std::vector<glm::vec3>& cp, float spacing, std::vector<glm::vec3>& res)
{
  // indices of the relevant control points
  int i0 = glm::clamp<int>(index - 1, 0, int(cp.size() - 1));
  int i1 = glm::clamp<int>(index, 0, int(cp.size() - 1));
  int i2 = glm::clamp<int>(index + 1, 0, int(cp.size() - 1));
  int i3 = glm::clamp<int>(index + 2, 0, int(cp.size() - 1));

  glm::vec3 from = cp[i1];
  glm::vec3 to   = cp[i2];

  float len = glm::length(from - to);

  uint32_t samples = uint32_t(ceil(len / spacing));

  float step = 1.f / samples;
  res.reserve(res.size() + samples);
  for(uint32_t i = 0; i < samples; i++)
  {
    res.push_back(glm::catmullRom(cp[i0], cp[i1], cp[i2], cp[i3], i * step));
  }
}


// Function to rotate and flip the coordinates based on Hilbert curve rules
void rot(int n, int* x, int* y, int rx, int ry)
{
  if(ry == 0)
  {
    if(rx == 1)
    {
      *x = n - 1 - *x;
      *y = n - 1 - *y;
    }
    // Swap x and y
    int temp = *x;
    *x       = *y;
    *y       = temp;
  }
}

// Function to convert t to a point on Hilbert curve at level n
glm::vec2 hilbertCurvePoint(float t, int n)
{
  int N = 1 << n;                       // 2^n, the dimension of the Hilbert curve grid
  int d = static_cast<int>(t * N * N);  // Map t to an integer index in the Hilbert curve

  int x = 0, y = 0;
  for(int s = 1; s < N; s *= 2)
  {
    int rx = (d / 2) % 2;
    int ry = (d ^ rx) % 2;
    rot(s, &x, &y, rx, ry);
    x += s * rx;
    y += s * ry;
    d /= 4;
  }

  return glm::vec2(static_cast<float>(x) / N, static_cast<float>(y) / N);
}


uint32_t PartitionedTlasSample::computeDominoCount()
{
  // Instances
  std::vector<glm::vec3> cp = generateControlPoints();


  uint32_t targetSize = 0;
  for(size_t i = 0; i < cp.size() - 1; i++)
  {
    targetSize += computeSegmentRequirements(uint32_t(i), cp, m_dominoHeight * m_dominoSpacing);
  }

  return targetSize;
}


std::vector<glm::vec3> PartitionedTlasSample::generateControlPoints()
{
  std::vector<glm::vec3> cp;

  float lengthMultiplier = m_sceneSize;

  uint32_t cpCount = uint32_t(m_sceneSize * 4);

  for(uint32_t i = 0; i < cpCount; i++)
  {
    float t = (float(i) + 0.5f) / float(cpCount);


    glm::vec2 coord = hilbertCurvePoint(t, 8) * 2.f - 1.f;

    cp.push_back(glm::vec3(lengthMultiplier * coord.x, lengthMultiplier * coord.y, 0.f));
  }

  return cp;
}


class UniformBitGrid
{
public:
  inline void create(const glm::uvec2& resolution, bool defaultValue)
  {
    m_resolution = resolution;

    uint32_t size = (resolution.x * resolution.y + 7) / 8;
    m_data        = new uint8_t[size];
    memset(m_data, defaultValue ? 0xFF : 0x00, size);
  }

  inline void destroy() { delete[] m_data; }

  inline void set(const glm::uvec2& coord, bool value)
  {
    if(coord.x >= m_resolution.x || coord.y >= m_resolution.y)
      return;

    uint32_t index  = coord.x + coord.y * m_resolution.x;
    uint32_t entry  = index / 8;
    uint32_t offset = index % 8;

    uint8_t byte = value ? (1 << offset) : 0x00;

    m_data[entry] = (m_data[entry] & ~byte) | byte;
  }

  inline bool get(const glm::uvec2& coord) const
  {
    if(coord.x >= m_resolution.x || coord.y >= m_resolution.y)
      return true;
    uint32_t index  = coord.x + coord.y * m_resolution.x;
    uint32_t entry  = index / 8;
    uint32_t offset = index % 8;

    return (((m_data[entry] >> offset) & 0x1) != 0);
  }
  inline const glm::uvec2& getResolution() const { return m_resolution; }

private:
  uint8_t*   m_data{};
  glm::uvec2 m_resolution{};
};

void PartitionedTlasSample::createScene(bool resetCamera)
{
  nvh::ScopedTimer st(__FUNCTION__);

  m_staticObjectCount  = 0u;
  m_dynamicObjectCount = 0u;

  float lengthMultiplier = m_sceneSize;

  m_animationShaderData.dynamicDistanceThreshold = m_sceneSize / 10.f;

  std::vector<glm::vec3> controlPoints = generateControlPoints();


  UniformBitGrid grid;

  float partitionSize = 2.f * lengthMultiplier / float(m_partitionCountPerAxis);
  float tileSize      = 3.f;

  float tilesPerPartition = partitionSize / tileSize;
  float integerTileCount  = floor(tilesPerPartition);

  float overshoot = (tilesPerPartition - integerTileCount) * tileSize;
  tileSize += overshoot / integerTileCount;

  grid.create({1.f + 2.f * lengthMultiplier / tileSize, 1.f + 2.f * lengthMultiplier / tileSize}, false);

#ifdef MAX_INSTANCES_PER_PARTITION
  std::vector<uint32_t> instanceCountPerPartition(m_partitionCountPerAxis * m_partitionCountPerAxis);
  uint32_t              skipCounter = 0u;
#endif
  std::vector<nvh::Node> dynamicNodes;
  uint32_t               archesSpacing;
  uint32_t               archesCount;

  std::vector<glm::uvec4> archesLocations;

  {
    std::vector<glm::vec3> dynamicSamples;
    {
      uint32_t targetSize = 0;
      for(size_t i = 0; i < controlPoints.size() - 1; i++)
      {
        targetSize += computeSegmentRequirements(uint32_t(i), controlPoints, m_dominoHeight * m_dominoSpacing);
      }
      dynamicSamples.reserve(targetSize);
    }

    {
      for(size_t i = 0; i < controlPoints.size() - 1; i++)
      {
        sampleSegment(uint32_t(i), controlPoints, m_dominoHeight * m_dominoSpacing, dynamicSamples);
      }
    }


    LOGI("Placing %zu dominoes\n", dynamicSamples.size());
    m_dominoCount = uint32_t(dynamicSamples.size());
    m_meshes.emplace_back(nvh::createCube());


    const glm::vec3 freq = glm::vec3(1.33333F, 2.33333F, 3.33333F) * static_cast<float>(0);
    const glm::vec3 v    = static_cast<glm::vec3>(sin(freq) * 0.5F + 0.5F);
    m_materials.push_back({glm::vec4(v, 1)});

    dynamicNodes.reserve(dynamicSamples.size());

    archesSpacing = 10;
    archesCount   = uint32_t(dynamicSamples.size()) / archesSpacing;

    archesLocations.reserve(archesCount);


    {
      glm::vec3 lastPos = dynamicSamples[1];
      for(size_t i = 0; i < dynamicSamples.size(); i++)
      {

        glm::vec3 pos = dynamicSamples[i];

#ifdef MAX_INSTANCES_PER_PARTITION
        uint32_t  partitionIndex = partitionIndexFromPosition(glm::vec3(pos.x, 0.f, pos.y));
        uint32_t& partitionSize  = instanceCountPerPartition[partitionIndex];
        if(partitionSize >= MAX_INSTANCES_PER_PARTITION)
        {
          skipCounter++;
          continue;
        }
        else
        {
          instanceCountPerPartition[partitionIndex]++;
        }
#endif
        nvh::Node& n    = dynamicNodes.emplace_back();
        n.mesh          = int32_t(m_meshes.size() - 1);
        n.material      = int32_t(m_materials.size() - 1);
        n.translation.x = pos.x;
        n.translation.y = m_dominoHeight * 0.5f;
        n.translation.z = pos.y;

        m_dynamicObjectCount++;


        float normX = glm::clamp((0.5f + (0.5f * pos.x / lengthMultiplier)), 0.f, 1.f) * grid.getResolution().x;
        float normY = glm::clamp((0.5f + (0.5f * pos.y / lengthMultiplier)), 0.f, 1.f) * grid.getResolution().y;

        uint32_t gridPosX = uint32_t(normX);
        uint32_t gridPosY = uint32_t(normY);

        for(int32_t y = -1; y <= 1; y++)
        {
          for(int32_t x = -1; x <= 1; x++)
          {
            grid.set({gridPosX + x, gridPosY + y}, true);
          }
        }

        glm::vec2 derivative = {pos.x - lastPos.x, pos.y - lastPos.y};

        if(i == 0)
        {
          derivative = -derivative;
        }

        derivative /= float(controlPoints.size() - 1) / dynamicSamples.size();

        n.scale    = m_dominoHeight * glm::vec3(0.1f, 1.f, 0.5f);
        n.rotation = computeQuaternion(derivative);

        lastPos = pos;


        glm::vec2 normal = {derivative.y, -derivative.x};
        normal           = glm::normalize(normal);
        if(i % archesSpacing == 0)
        {
          float      archSize = 2.5f;
          glm::uvec4 arch     = {gridPosX - uint32_t(archSize * normal.x), gridPosY - uint32_t(archSize * normal.y),
                                 gridPosX + uint32_t(archSize * normal.x), gridPosY + uint32_t(archSize * normal.y)};
          archesLocations.emplace_back(arch);
        }
      }
    }

    m_animationShaderData.objectBboxMin = glm::vec3(-lengthMultiplier, 0.f, -lengthMultiplier);
    m_animationShaderData.objectBboxMax = glm::vec3(lengthMultiplier, 0.f, lengthMultiplier);
  }


  {
    if(true)
    {
      // Adding an underlying plane to make sure there are no holes in the scene
      m_materials.push_back({glm::vec4(.7F, .7F, .7F, 1.0F)});


      m_meshes.emplace_back(nvh::createPlane(10, partitionSize, partitionSize));
      for(int32_t y = 0; y < m_partitionCountPerAxis; y++)
      {
        for(int32_t x = 0; x < m_partitionCountPerAxis; x++)
        {

          glm::vec3 pos;
          pos.x = -lengthMultiplier + x * partitionSize + (partitionSize / 2.f);
          pos.y = -0.1f;
          pos.z = -lengthMultiplier + y * partitionSize + (partitionSize / 2.f);

#ifdef MAX_INSTANCES_PER_PARTITION
          uint32_t partitionIndex = partitionIndexFromPosition(pos);
          if(instanceCountPerPartition[partitionIndex] >= MAX_INSTANCES_PER_PARTITION)
          {
            skipCounter++;
            continue;
          }
          else
          {
            instanceCountPerPartition[partitionIndex]++;
          }
#endif
          nvh::Node& n  = m_nodes.emplace_back();
          n.mesh        = static_cast<int>(m_meshes.size()) - 1;
          n.material    = static_cast<int>(m_materials.size()) - 1;
          n.translation = pos;
        }
      }
    }


    std::random_device                                       dev;
    std::mt19937                                             rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> distGrid(0, grid.getResolution().x - 1);
    std::uniform_real_distribution<float>                    distFloat(0.f, 1.f);


    // Semi-transparent icosahedra
    if(true)
    {
      m_meshes.emplace_back(nvh::createIcosahedron());
      m_materials.push_back({glm::vec4(.4F, .7F, .4F, 0.4F)});

      uint32_t objectCount = (grid.getResolution().x * grid.getResolution().y) / 100;
      m_nodes.reserve(m_nodes.size() + objectCount);
      for(uint32_t i = 0; i < objectCount; i++)
      {


        glm::uvec2 upos = {0, 0};

        while(grid.get(upos))
        {
          upos = {distGrid(rng), distGrid(rng)};
        }


        glm::vec2 posNoise = {0.f + 0.4f * (distFloat(rng) - 0.5f), 0.f + 0.4f * (distFloat(rng) - 0.5f)};

        glm::vec2 pos = ((2.f * (glm::vec2(upos) + posNoise) / glm::vec2(grid.getResolution())) - glm::vec2(1.f)) * lengthMultiplier;

#ifdef MAX_INSTANCES_PER_PARTITION
        {
          uint32_t partitionIndex = partitionIndexFromPosition(glm::vec3(pos.x, 0.f, pos.y));
          if(instanceCountPerPartition[partitionIndex] >= MAX_INSTANCES_PER_PARTITION)
          {
            skipCounter++;
            continue;
          }
          else
          {
            instanceCountPerPartition[partitionIndex]++;
          }
        }
#endif

        grid.set(upos, true);


        nvh::Node& n = m_nodes.emplace_back();
        n.mesh       = int32_t(m_meshes.size() - 1);
        n.material   = int32_t(m_materials.size() - 1);

        n.translation.x = pos.x;
        n.translation.z = pos.y;

        float heightVariation = 0.2f + 3.f * (distFloat(rng));

        n.translation.y     = heightVariation / 2.f;
        n.scale             = glm::vec3(heightVariation);
        float rotationAngle = M_PI * 0.5f;
        n.rotation          = glm::angleAxis(rotationAngle, glm::vec3{1.f, 0.f, 1.f});
        rotationAngle       = distFloat(rng) * M_PI * 1.f;
        n.rotation          = glm::angleAxis(rotationAngle, glm::vec3{0.f, 1.f, 0.f});
      }
    }

    // Tiles with variable height
    if(true)
    {
      float tileHeight = 0.1f;
      m_meshes.emplace_back(nvh::createCube(tileSize * 0.98f, 0.1f, tileSize * 0.98f));
      m_materials.push_back({glm::vec4(.5F, .5F, .5F, 1.0F)});
      m_nodes.reserve(m_nodes.size() + grid.getResolution().x * grid.getResolution().y);
      for(uint32_t y = 0; y < grid.getResolution().y; y++)
      {
        for(uint32_t x = 0; x < grid.getResolution().x; x++)
        {


          glm::uvec2 upos = {x, y};


          glm::vec2 posNoise = {0.f + 0.01f * (distFloat(rng) - 0.0f), 0.f + 0.01f * (distFloat(rng) - 0.0f)};

          glm::vec2 pos = ((2.f * (glm::vec2(upos) + posNoise) / glm::vec2(grid.getResolution())) - glm::vec2(1.f)) * lengthMultiplier;

#ifdef MAX_INSTANCES_PER_PARTITION
          {
            uint32_t partitionIndex = partitionIndexFromPosition(glm::vec3(pos.x, 0.f, pos.y));
            if(instanceCountPerPartition[partitionIndex] >= MAX_INSTANCES_PER_PARTITION)
            {
              skipCounter++;
              continue;
            }
            else
            {
              instanceCountPerPartition[partitionIndex]++;
            }
          }
#endif
          nvh::Node& n    = m_nodes.emplace_back();
          n.mesh          = int32_t(m_meshes.size() - 1);
          n.material      = int32_t(m_materials.size() - 1);
          n.translation.x = pos.x;
          n.translation.z = pos.y;
          n.translation.y = -tileHeight;
          if(!grid.get(upos))
          {
            float     rotationAngle = distFloat(rng) * M_PI * 0.1f;
            glm::vec3 rotationAxis  = {distFloat(rng), distFloat(rng), distFloat(rng)};

            n.rotation = glm::angleAxis(rotationAngle, glm::vec3{0.f, 1.f, 0.f}) * n.rotation;
            n.scale.y  = 1.f + distFloat(rng) * 5.f;
          }
          else
          {
            float     rotationAngle = distFloat(rng) * M_PI * 0.01f;
            glm::vec3 rotationAxis  = {distFloat(rng), distFloat(rng), distFloat(rng)};

            n.rotation = glm::angleAxis(rotationAngle, glm::vec3{0.f, 1.f, 0.f}) * n.rotation;
          }
        }
      }
    }

    // Arches
    if(true)
    {
      m_meshes.emplace_back(nvh::createTetrahedron());
      m_materials.push_back({glm::vec4(.5F, .5F, .5F, 1.0F)});
      m_nodes.reserve(m_nodes.size() + archesCount * 2);

      for(glm::uvec4 arch : archesLocations)
      {

        glm::uvec2 upos0 = {arch.x, arch.y};
        glm::uvec2 upos1 = {arch.z, arch.w};

        if(grid.get(upos0) || grid.get(upos1))
        {
          continue;
        }

        glm::vec2 tileSize = lengthMultiplier / glm::vec2(grid.getResolution());

        glm::vec2 pos0 = ((2.f * (glm::vec2(upos0)) / glm::vec2(grid.getResolution())) - glm::vec2(1.f)) * lengthMultiplier;
        grid.set(upos0, true);
        glm::vec2 pos1 = ((2.f * (glm::vec2(upos1)) / glm::vec2(grid.getResolution())) - glm::vec2(1.f)) * lengthMultiplier;
        grid.set(upos1, true);

        //float objectSize = tileSize.x / 10.f;
        float    objectSize  = tileSize.x / 3.f;
        uint32_t objectCount = tileSize.x / objectSize;

        float     archLength = glm::length(pos1 - pos0);
        glm::vec2 archDir    = glm::normalize(pos1 - pos0);
        glm::vec2 archNormal = glm::vec2(archDir.y, -archDir.x);

        uint32_t archSampleCount = uint32_t(archLength / objectSize);
        float    archHeight      = m_dominoHeight * 3.f;
        for(uint32_t archSample = 0; archSample < archSampleCount; archSample++)
        {
          float archRatio = float(archSample + 1) / archSampleCount;
          for(uint32_t idxOnLine = 0; idxOnLine < objectCount; idxOnLine++)
          {
            float lineRatio = float(idxOnLine + 1) / objectCount;

            glm::vec3 pos0Vec3;
            pos0Vec3.x = pos0.x - 0.5f * tileSize.x + idxOnLine * objectSize + archDir.x * 0.5f * archLength * archRatio
                         + 2.f * (lineRatio - 0.5f) * sinf(archRatio * M_PI * 0.5f) * archNormal.x * archHeight;
            pos0Vec3.z = pos0.y - 0.5f * tileSize.y + idxOnLine * objectSize + archDir.y * 0.5f * archLength * archRatio
                         + 2.f * (lineRatio - 0.5f) * sinf(archRatio * M_PI * 0.5f) * archNormal.y * archHeight;
            pos0Vec3.y = 0.1f + sinf(archRatio * M_PI * 0.5f) * archHeight;

#ifdef MAX_INSTANCES_PER_PARTITION
            {
              uint32_t partitionIndex = partitionIndexFromPosition(pos0Vec3);
              if(instanceCountPerPartition[partitionIndex] >= MAX_INSTANCES_PER_PARTITION)
              {
                skipCounter++;
                continue;
              }
              else
              {
                instanceCountPerPartition[partitionIndex]++;
              }
            }
#endif
            nvh::Node& n0 = m_nodes.emplace_back();
            n0.mesh       = int32_t(m_meshes.size() - 1);
            n0.material   = int32_t(m_materials.size() - 1);


            n0.scale       = glm::vec3(objectSize, objectSize, objectSize);
            n0.rotation    = glm::angleAxis(float(archRatio * M_PI * 0.5f), glm::vec3{0.f, 1.f, 0.f});
            n0.translation = pos0Vec3;


            glm::vec3 pos1Vec3;
            pos1Vec3.x = pos1.x - 0.5f * tileSize.x + idxOnLine * objectSize - archDir.x * 0.5f * archLength * archRatio
                         + 2.f * (lineRatio - 0.5f) * sinf(archRatio * M_PI * 0.5f) * archNormal.x * archHeight;
            pos1Vec3.z = pos1.y - 0.5f * tileSize.y + idxOnLine * objectSize - archDir.y * 0.5f * archLength * archRatio
                         + 2.f * (lineRatio - 0.5f) * sinf(archRatio * M_PI * 0.5f) * archNormal.y * archHeight;

            pos1Vec3.y = 0.1f + sinf(archRatio * M_PI * 0.5f) * archHeight;

#ifdef MAX_INSTANCES_PER_PARTITION
            {
              uint32_t partitionIndex = partitionIndexFromPosition(pos1Vec3);
              if(instanceCountPerPartition[partitionIndex] >= MAX_INSTANCES_PER_PARTITION)
              {
                skipCounter++;
                continue;
              }
              else
              {
                instanceCountPerPartition[partitionIndex]++;
              }
            }
#endif
            nvh::Node& n1 = m_nodes.emplace_back();
            n1.mesh       = int32_t(m_meshes.size() - 1);
            n1.material   = int32_t(m_materials.size() - 1);


            n1.scale       = glm::vec3(objectSize, objectSize, objectSize);
            n1.rotation    = glm::angleAxis(float(archRatio * M_PI * 0.5f), glm::vec3{0.f, 1.f, 0.f});
            n1.translation = pos1Vec3;
          }
        }
      }
    }
  }

  m_staticObjectCount = uint32_t(m_nodes.size());

  // The dynamic nodes are expected at the end of the nodes vector
  {
    m_nodes.insert(m_nodes.end(), dynamicNodes.begin(), dynamicNodes.end());
  }

#ifdef MAX_INSTANCES_PER_PARTITION
  if(skipCounter > 0)
  {
    LOGW("Skipped %d instances due to MAX_INSTANCES_PER_PARTITION\n", skipCounter);
  }
#endif

  m_animationShaderData.totalObjectCount = uint32_t(m_nodes.size());

  // Setting camera to see the scene
  if(resetCamera)
  {
    CameraManip.setClipPlanes({0.1F, 100.0F});
    CameraManip.setLookat({-20.F, 10.0F, 15.0F}, {-15.F, 7.F, 7.0F}, {0.0F, 1.0F, 0.0F});
    CameraManip.setMode(nvh::CameraManipulator::Fly);
    CameraManip.setSpeed(20.f);
  }
  // Default parameters for overall material
  m_pushConst.intensity = 5.0F;
  m_pushConst.maxDepth  = MAXRAYRECURSIONDEPTH;
  m_pushConst.roughness = 1.0F;
  m_pushConst.metallic  = 0.1F;

  // Default Sky values
  m_skyParams = nvvkhl_shaders::initSimpleSkyParameters();
}


// Create all Vulkan buffer data
void PartitionedTlasSample::createVkBuffers()
{
  nvh::ScopedTimer st(__FUNCTION__);

  VkCommandBuffer cmd = m_app->createTempCmdBuffer();
  m_bMeshes.resize(m_meshes.size());

  const VkBufferUsageFlags rt_usage_flag = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                           | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

  // Create a buffer of Vertex and Index per mesh
  for(size_t i = 0; i < m_meshes.size(); i++)
  {
    PrimitiveMeshVk& m = m_bMeshes[i];
    m.vertices         = m_alloc->createBuffer(cmd, m_meshes[i].vertices, rt_usage_flag);
    m.indices          = m_alloc->createBuffer(cmd, m_meshes[i].triangles, rt_usage_flag);
    m_dutil->DBG_NAME_IDX(m.vertices.buffer, i);
    m_dutil->DBG_NAME_IDX(m.indices.buffer, i);
  }

  // Create the buffer of the current frame, changing at each frame
  m_bFrameInfo = m_alloc->createBuffer(sizeof(DH::FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  m_dutil->DBG_NAME(m_bFrameInfo.buffer);

  // Create the buffer of sky parameters, updated at each frame
  m_bSkyParams = m_alloc->createBuffer(sizeof(nvvkhl_shaders::SimpleSkyParameters), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  m_dutil->DBG_NAME(m_bSkyParams.buffer);

  // Primitive instance information
  std::vector<DH::InstanceInfo> inst_info;
  inst_info.resize(m_nodes.size());

  uint32_t numThreads = std::min((uint32_t)m_nodes.size(), std::thread::hardware_concurrency());
  nvh::parallel_batches(
      m_nodes.size(),
      [&](uint64_t i) {
        nvh::Node&       node = m_nodes[i];
        DH::InstanceInfo info{.transform = node.localMatrix(), .materialID = node.material, .meshID = node.mesh};
        inst_info[i] = info;
      },
      numThreads);


  m_bInstInfoBuffer =
      m_alloc->createBuffer(cmd, inst_info, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  m_dutil->DBG_NAME(m_bInstInfoBuffer.buffer);

  m_bMaterials = m_alloc->createBuffer(cmd, m_materials, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  m_dutil->DBG_NAME(m_bMaterials.buffer);

  m_app->submitAndWaitTempCmdBuffer(cmd);
}


// Initialize data for the physics simulation
void PartitionedTlasSample::createAnimationData()
{
  m_animationShaderData.resetToOriginal  = 1;
  m_animationShaderData.toppleStrength   = 0.f;
  m_animationShaderData.instancesAddress = m_instancesBuffer.address;
  m_animationShaderData.ptlasActive      = 1;

  m_partitionIndexPerNode.resize(m_dynamicObjectCount + m_staticObjectCount);

  m_animationShaderData.dynamicObjectCount = uint32_t(m_dynamicObjectCount);
  std::vector<DH::AnimationState> originalState(m_dynamicObjectCount);

  uint32_t numThreads = std::min((uint32_t)m_nodes.size(), std::thread::hardware_concurrency());
  nvh::parallel_batches(
      m_dynamicObjectCount,
      [&](uint64_t i) {
        originalState[i].linearVelocity = glm::vec3(0.f);
        DH::setTransform4x4(originalState[i], m_nodes[i + m_staticObjectCount].localMatrix());
        originalState[i].stateID = STATE_FREE;
      },
      numThreads);


  glm::vec3 bboxMin  = m_animationShaderData.objectBboxMin;
  glm::vec3 bboxMax  = m_animationShaderData.objectBboxMax;
  glm::vec3 bboxSize = bboxMax - bboxMin;

  m_animationShaderData.partitionsPerAxis = m_partitionCountPerAxis;

  size_t selfToppleInterval =
      (m_targetToppleSeedCount == 0) ?
          ~size_t(0) :
          ((m_targetToppleSeedCount < 0) ? 10000ull : size_t(floor(m_dynamicObjectCount / m_targetToppleSeedCount)));
  selfToppleInterval                   = std::max(size_t(1), selfToppleInterval);
  std::atomic_uint32_t selfToppleCount = 0u;

  std::vector<uint32_t> perPartitionCounter(m_partitionCountPerAxis * m_partitionCountPerAxis);


  nvh::parallel_batches(
      m_dynamicObjectCount,
      [&](uint64_t i) {
        size_t globalIndex = m_staticObjectCount + i;

        glm::vec3 originalPosition = DH::getPosition(originalState[i]);

        // The partitions are defined as a simple ground-aligned grid
        originalState[i].partitionID = partitionIndexFromPosition(originalPosition);  // indexX + m_partitionCountPerAxis * indexY;
        originalState[i].newPartitionID = originalState[i].partitionID;

        m_partitionIndexPerNode[globalIndex] = originalState[i].partitionID;
        originalState[i].lastContact         = ~0u;
        originalState[i].firstContact        = ~0u;
        originalState[i].lastModified        = 0;
        // Seed the animation by toppling a target number of dominoes
        if(selfToppleInterval != ~size_t(0) && i % selfToppleInterval == 0 && selfToppleCount < uint32_t(m_targetToppleSeedCount))
        {
          selfToppleCount++;
          glm::vec3 orientation;
          if(i < m_dynamicObjectCount - 1)
          {
            orientation = glm::normalize(DH::getPosition(originalState[i + 1]) - DH::getPosition(originalState[i]));
          }
          else
          {
            orientation = -glm::normalize(DH::getPosition(originalState[i]) - DH::getPosition(originalState[i - 1]));
          }

          glm::vec3 axis = normalize(glm::cross(glm::vec3(0.f, 1.f, 0.f), orientation));

          originalState[i].angularVelocity = -axis * 5.5f;
        }
        else
        {
          originalState[i].angularVelocity = glm::vec3(0.f);
        }
      },
      numThreads);

  // Adding 1 for the global partition
  std::vector<DH::PartitionState> partitionState(m_partitionCountPerAxis * m_partitionCountPerAxis + 1);

  std::vector<std::atomic_uint32_t> staticCounter(m_partitionCountPerAxis * m_partitionCountPerAxis + 1);

  nvh::parallel_batches(
      m_staticObjectCount,
      [&](uint64_t i) {
        size_t globalIndex = i;

        glm::vec3 originalPosition = m_nodes[globalIndex].localMatrix()[3];

        // The partitions are defined as a simple ground-aligned grid
        uint32_t partitionID                 = partitionIndexFromPosition(originalPosition);
        m_partitionIndexPerNode[globalIndex] = partitionID;
        staticCounter[partitionID].fetch_add(1, std::memory_order_relaxed);
      },
      numThreads);

  for(size_t i = 0; i < partitionState.size(); i++)
  {
    partitionState[i].staticObjectCount = staticCounter[i].load(std::memory_order_relaxed);
  }

  std::vector<DH::AnimationGlobalState> globalState(1);
  globalState[0].currentCollisionIndex = 0;
  globalState[0].toppleRequest         = ~0u;
  globalState[0].focus                 = ~0u;


  for(auto& p : partitionState)
  {
    p.lastModified = ~0u;
  }

  {
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    m_originalState     = m_alloc->createBuffer(cmd, originalState, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    m_state[0] = m_alloc->createBuffer(cmd, originalState, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    m_state[1] = m_alloc->createBuffer(cmd, originalState, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    m_globalState = m_alloc->createBuffer(cmd, globalState, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    m_globalStateHost = m_alloc->createBuffer(cmd, globalState, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    m_partitionState = m_alloc->createBuffer(cmd, partitionState, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

#ifdef USE_NVVK_INSPECTOR
    nvvkhl::ElementInspector::BufferInspectionInfo bufferInfo{};
    bufferInfo.entryCount   = originalState.size();
    bufferInfo.format       = g_elementInspector->formatStruct(DH::getInspectorStringAnimationState());
    bufferInfo.name         = "State0";
    bufferInfo.sourceBuffer = m_state[0].buffer;
    g_elementInspector->initBufferInspection(eState0, bufferInfo);

    bufferInfo.name         = "State1";
    bufferInfo.sourceBuffer = m_state[1].buffer;
    g_elementInspector->initBufferInspection(eState1, bufferInfo);

    bufferInfo.name         = "StateOriginal";
    bufferInfo.sourceBuffer = m_originalState.buffer;
    g_elementInspector->initBufferInspection(eStateOriginal, bufferInfo);


    bufferInfo.entryCount   = partitionState.size();
    bufferInfo.format       = g_elementInspector->formatStruct(DH::getInspectorStringPartitionState());
    bufferInfo.name         = "Partition State";
    bufferInfo.sourceBuffer = m_partitionState.buffer;
    g_elementInspector->initBufferInspection(ePartitionState, bufferInfo);
#endif
    m_app->submitAndWaitTempCmdBuffer(cmd);
  }

  m_animationShaderData.originalState = m_originalState.address;
  m_animationShaderData.state[0]      = m_state[0].address;
  m_animationShaderData.state[1]      = m_state[1].address;

  m_animationShaderData.currentStateIndex = 0;
  m_animationShaderData.globalState       = m_globalState.address;
  m_animationShaderData.partitionState    = m_partitionState.address;
  m_animationShaderData.frameIndex        = 0;

  recompileAuxShaders();
}
