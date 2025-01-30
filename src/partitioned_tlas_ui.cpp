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
#include "imgui/imgui_icon.h"


void tooltip(const std::string& text)
{
  if(ImGui::IsItemHovered() && ImGui::BeginTooltip())
  {
    ImGui::Text("%s", text.c_str());
    ImGui::EndTooltip();
  }
}

void PartitionedTlasSample::onResize(uint32_t width, uint32_t height)
{
  m_gBuffers = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(),
                                                 VkExtent2D{m_sizeMultiplier * width, m_sizeMultiplier * height},
                                                 std::vector<VkFormat>({m_colorFormat, m_colorFormat, VK_FORMAT_R32_SFLOAT}));
  writeRtDesc();

  m_compositingDispatcher.updateBinding(B_outImage, m_gBuffers->getColorImageView(0), VK_IMAGE_LAYOUT_GENERAL);
  m_compositingDispatcher.updateBinding(B_aoImage, m_gBuffers->getColorImageView(1), VK_IMAGE_LAYOUT_GENERAL);
  m_compositingDispatcher.updateBinding(B_depthImage, m_gBuffers->getColorImageView(2), VK_IMAGE_LAYOUT_GENERAL);
}

void PartitionedTlasSample::onUIRender()
{

  {  // Setting menu
    ImGui::Begin("Settings");

    if(!hasValidAnimationShaders() || m_invalidShaderNotified)
    {
      ImGui::TextColored(ImVec4(1.f, 0.f, 0.f, 1.f), "Shader compilation error(s) - see log for details");
    }


    ImGui::Text("Right-click on a domino to topple it");
    ImGui::PushFont(ImGuiH::getIconicFont());

    if(m_run)
    {
      if(ImGui::Button(ImGuiH::icon_media_pause, ImVec2(32, 32)))
      {
        m_run = false;
      }
    }
    else
    {
      if(ImGui::Button(ImGuiH::icon_media_play, ImVec2(32, 32)))
      {
        m_run = true;
      }
    }


    ImGui::PopFont();
    if(ImGui::IsItemHovered() && ImGui::BeginTooltip())
    {

      ImGui::Text("Play/pause the physics");
      ImGui::EndTooltip();
    }


    ImGui::SameLine();

    ImGui::PushFont(ImGuiH::getIconicFont());
    if(ImGui::Button(ImGuiH::icon_media_stop, ImVec2(32, 32)))
    {
      vkDeviceWaitIdle(m_device);
      m_animationShaderData.resetToOriginal = 1;
      m_step                                = true;
      m_run                                 = false;
    }
    ImGui::PopFont();
    tooltip("Stop physics & reset the dominoes");

    ImGui::SameLine();
    ImGui::PushFont(ImGuiH::getIconicFont());
    if(ImGui::Button(ImGuiH::icon_media_step_forward, ImVec2(32, 32)))
    {
      m_step = true;
    }
    ImGui::PopFont();
    tooltip("Step physics");


    if(ImGui::CollapsingHeader("Scene", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
    {

      ImGui::Text("Current domino count: %d", m_dominoCount);
      ImGui::Text("Current static object count: %d", m_staticObjectCount);

      ImGuiH::PropertyEditor::begin();

      int32_t oldPartitionCountPerAxis = m_partitionCountPerAxis;
      ImGuiH::PropertyEditor::InputInt("Partitions per axis", &m_partitionCountPerAxis);
      m_partitionCountPerAxis = std::max(1, m_partitionCountPerAxis);
      float oldSceneSize      = m_sceneSize;
      ImGuiH::PropertyEditor::InputFloat("Scene size", &m_sceneSize);
      m_sceneSize = std::max(1.f, m_sceneSize);

      int32_t oldTargetToppleSeedCount = m_targetToppleSeedCount;
      ImGuiH::PropertyEditor::InputInt("Self-toppling dominoes", &m_targetToppleSeedCount);
      m_targetToppleSeedCount = std::max(0, m_targetToppleSeedCount);


      ImGuiH::PropertyEditor::end();

      bool propertiesWereJustChanged = oldSceneSize != m_sceneSize || oldPartitionCountPerAxis != m_partitionCountPerAxis
                                       || oldTargetToppleSeedCount != m_targetToppleSeedCount;

      m_scenePropertiesChanged = m_scenePropertiesChanged || propertiesWereJustChanged;

      if(propertiesWereJustChanged || m_generateDominoCount == 0u)
      {
        m_generateDominoCount = computeDominoCount();
      }

      ImGui::BeginDisabled(!m_scenePropertiesChanged);

      ImGui::Text("New domino count: %d", m_generateDominoCount);

      if(ImGui::Button("Regenerate scene"))
      {
        resetScene();
        m_scenePropertiesChanged = false;
      }
      ImGui::EndDisabled();
    }


    if(ImGui::CollapsingHeader("Acceleration Structure", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
    {


      if(ImGui::Checkbox("PTLAS Active", reinterpret_cast<bool*>(&m_animationShaderData.ptlasActive)))
      {
        vkDeviceWaitIdle(m_device);
        writeRtDesc();
        m_animationShaderData.resetToOriginal = 1;
      }

      ImGui::Separator();
      ImGui::TreePush("PTLAS");

      ImGui::BeginDisabled(m_animationShaderData.ptlasActive == 0);
      ImGui::Text("PTLAS");
      ImGui::Text("Partition update mode");
      ImGui::TreePush("UpdateMode");
      ImGui::RadioButton("Always update partition", &m_animationShaderData.dynamicUpdateMode, PTLAS_DYNAMIC_ALWAYS_UPDATE);
      tooltip("When a domino moves, rewrite its instance and update the partition it belongs to");

      ImGui::RadioButton("Always move dynamic to global", &m_animationShaderData.dynamicUpdateMode, PTLAS_DYNAMIC_MOVE_TO_GLOBAL);
      tooltip("When a domino moves, remove it from its partition and write it into the global partition, where it will remain until its motion stops. ");

      ImGui::RadioButton("Update partition nearby, move to global otherwise", &m_animationShaderData.dynamicUpdateMode,
                         PTLAS_DYNAMIC_UPDATE_OR_MOVE_TO_GLOBAL);
      tooltip("When a domino moves, if its partition center is below the threshold distance, update its partition. \n Otherwise remove it from its partition and write it into the global partition, where it will remain until its motion stops. ");
      ImGui::TreePop();

      bool useGlobalPartition = m_animationShaderData.dynamicUpdateMode == PTLAS_DYNAMIC_MOVE_TO_GLOBAL
                                || m_animationShaderData.dynamicUpdateMode == PTLAS_DYNAMIC_UPDATE_OR_MOVE_TO_GLOBAL;

      ImGui::BeginDisabled(!useGlobalPartition);
      ImGui::Checkbox("Mark all dominoes dynamic in partition", (bool*)&m_animationShaderData.dynamicMarkAllDominos);
      tooltip("When a domino starts toppling, move all other dominoes in its partition into the global partition");
      ImGui::EndDisabled();
      if(!useGlobalPartition)
      {
        m_animationShaderData.dynamicMarkAllDominos = 0;
      }


      ImGui::BeginDisabled(m_animationShaderData.dynamicUpdateMode != PTLAS_DYNAMIC_UPDATE_OR_MOVE_TO_GLOBAL);
      ImGui::InputFloat("Mode change distance", &m_animationShaderData.dynamicDistanceThreshold);
      tooltip("If the dynamic partition behavior depends on the partition location, \n defines the distance after which toppling dominoes are moved to the global partition");
      ImGui::EndDisabled();


      ImGui::EndDisabled();
      ImGui::TreePop();

      ImGui::Separator();
      ImGui::TreePush("TLAS");

      ImGui::BeginDisabled(m_animationShaderData.ptlasActive != 0);
      ImGui::Text("TLAS");
      ImGui::Checkbox("Refit TLAS", &m_tlasRefit);

      ImGui::EndDisabled();
      ImGui::TreePop();

      ImGui::TreePush("Stats");
      if(ImGui::CollapsingHeader("Statistics", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
      {

        ImGui::Text("TLAS : %.2f MB", m_stats.tlasAccelerationStructureSize / (1024.f * 1024.f));
        ImGui::Text("TLAS scratch: %.2f MB", m_stats.tlasBuildScratchSize / (1024.f * 1024.f));

        ImGui::Text("PTLAS : %.2f MB", m_stats.ptlasAccelerationStructureSize / (1024.f * 1024.f));
        ImGui::Text("PTLAS scratch: %.2f MB", m_stats.ptlasBuildScratchSize / (1024.f * 1024.f));


        ImGui::Separator();
        auto* globalState = reinterpret_cast<DH::AnimationGlobalState*>(m_alloc->map(m_globalStateHost));
        ImGui::Text("Updated instances: %d", globalState->instanceUpdateCount);
        m_alloc->unmap(m_globalStateHost);
      }
      ImGui::TreePop();
    }

    ImGui::Separator();
    if(ImGui::CollapsingHeader("Camera"))
    {
      ImGuiH::CameraWidget();
    }


    if(ImGui::CollapsingHeader("Lighting"))
    {
      ImGui::Text("Sun Orientation");
      ImGuiH::PropertyEditor::begin();
      glm::vec3 dir = m_skyParams.directionToLight;
      ImGuiH::azimuthElevationSliders(dir, false);
      m_skyParams.directionToLight = dir;
      ImGuiH::PropertyEditor::end();
    }


    ImGui::End();
  }
  {  // Rendering Viewport
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
    ImGui::Begin("Viewport");
    const auto corner = ImGui::GetCursorScreenPos();  // Corner of the viewport
    // Display the G-Buffer image
    ImGui::Image(m_gBuffers->getDescriptorSet(), ImGui::GetContentRegionAvail());

    if(ImGui::IsItemHovered())
    {
      const auto mousePos = ImGui::GetMousePos();

      m_animationShaderData.mouseCoord = {m_sizeMultiplier * (mousePos.x - corner.x),
                                          m_sizeMultiplier * (mousePos.y - corner.y)};
    }
    if(ImGui::IsMouseReleased(ImGuiMouseButton_Right))
    {
      m_animationShaderData.toppleDomino = 1;
      m_run                              = true;
    }
    else
    {
      m_animationShaderData.toppleDomino = 0;
    }

    if(ImGui::IsKeyReleased(ImGuiKey_R))
    {

      recompilePipeline();
      recompileAuxShaders();
    }


    {  // Display orientation axis at the bottom left corner of the window
      float  axisSize = 25.F;
      ImVec2 pos      = ImGui::GetWindowPos();
      pos.y += ImGui::GetWindowSize().y;
      pos += ImVec2(axisSize * 1.1F, -axisSize * 1.1F) * ImGui::GetWindowDpiScale();  // Offset
      ImGuiH::Axis(pos, CameraManip.getMatrix(), axisSize);
    }

    ImGui::End();
    ImGui::PopStyleVar();
  }
}
