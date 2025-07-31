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
#include "nvgui/fonts.hpp"
#include "nvgui/property_editor.hpp"
#include "nvgui/camera.hpp"
#include "nvgui/azimuth_sliders.hpp"
#include "nvgui/axis.hpp"


void tooltip(const std::string& text)
{
  if(ImGui::IsItemHovered() && ImGui::BeginTooltip())
  {
    ImGui::Text("%s", text.c_str());
    ImGui::EndTooltip();
  }
}

void PartitionedTlasSample::onResize(VkCommandBuffer cmd, const VkExtent2D& size)
{

  {
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    m_gBuffers.update(cmd, VkExtent2D{(uint32_t)m_sizeMultiplier * size.width, (uint32_t)m_sizeMultiplier * size.height});
    m_stagingUploader.cmdUploadAppended(cmd);
    m_app->submitAndWaitTempCmdBuffer(cmd);
  }


  writeRaytracingDescriptors();

  writeCompositingDescriptors();
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
    ImGui::PushFont(nvgui::getIconicFont());

    if(m_run)
    {
      if(ImGui::Button(nvgui::icon_media_pause, ImVec2(32, 32)))
      {
        m_run = false;
      }
    }
    else
    {
      if(ImGui::Button(nvgui::icon_media_play, ImVec2(32, 32)))
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

    ImGui::PushFont(nvgui::getIconicFont());
    if(ImGui::Button(nvgui::icon_media_stop, ImVec2(32, 32)))
    {
      vkDeviceWaitIdle(m_device);
      m_animationShaderData.resetToOriginal = 1;
      m_step                                = true;
      m_run                                 = false;
    }
    ImGui::PopFont();
    tooltip("Stop physics & reset the dominoes");

    ImGui::SameLine();
    ImGui::PushFont(nvgui::getIconicFont());
    if(ImGui::Button(nvgui::icon_media_step_forward, ImVec2(32, 32)))
    {
      m_step = true;
    }
    ImGui::PopFont();
    tooltip("Step physics");


    if(ImGui::CollapsingHeader("Scene", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
    {

      ImGui::Text("Current domino count: %d", m_dominoCount);
      ImGui::Text("Current static object count: %d", m_staticObjectCount);

      nvgui::PropertyEditor::begin();

      int32_t oldPartitionCountPerAxis = m_partitionCountPerAxis;
      nvgui::PropertyEditor::InputInt("Partitions per axis", &m_partitionCountPerAxis);
      m_partitionCountPerAxis = std::max(1, m_partitionCountPerAxis);
      float oldSceneSize      = m_sceneSize;
      nvgui::PropertyEditor::InputFloat("Scene size", &m_sceneSize);
      m_sceneSize = std::max(1.f, m_sceneSize);

      int32_t oldTargetToppleSeedCount = m_targetToppleSeedCount;
      nvgui::PropertyEditor::InputInt("Self-toppling dominoes", &m_targetToppleSeedCount);
      m_targetToppleSeedCount = std::max(0, m_targetToppleSeedCount);


      nvgui::PropertyEditor::end();

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
        writeRaytracingDescriptors();
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
        auto* globalState = reinterpret_cast<shaderio::AnimationGlobalState*>(m_globalStateHost.mapping);
        ImGui::Text("Updated instances: %d", globalState->instanceUpdateCount);
      }
      ImGui::TreePop();
    }

    ImGui::Separator();
    if(ImGui::CollapsingHeader("Camera"))
    {
      nvgui::CameraWidget(g_cameraManipulator);
    }


    if(ImGui::CollapsingHeader("Lighting"))
    {
      ImGui::Text("Sun Orientation");
      nvgui::PropertyEditor::begin();
      glm::vec3 dir = m_skyParams.sunDirection;
      nvgui::azimuthElevationSliders(dir, false, true);
      m_skyParams.sunDirection = dir;
      nvgui::PropertyEditor::end();
    }


    ImGui::End();
  }
  {  // Rendering Viewport
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
    ImGui::Begin("Viewport");
    const ImVec2 corner = ImGui::GetCursorScreenPos();  // Corner of the viewport
    // Display the G-Buffer image
    ImGui::Image(m_gBuffers.getDescriptorSet(1), ImGui::GetContentRegionAvail());

    if(ImGui::IsItemHovered())
    {
      const ImVec2 mousePos = ImGui::GetMousePos();

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
      writeRaytracingDescriptors();
      writeCompositingDescriptors();
    }


    {  // Display orientation axis at the bottom left corner of the window
      float  axisSize = 25.F;
      ImVec2 pos      = ImGui::GetWindowPos();
      pos.y += ImGui::GetWindowSize().y;
      pos += ImVec2(axisSize * 1.1F, -axisSize * 1.1F) * ImGui::GetWindowDpiScale();  // Offset
      nvgui::Axis(pos, g_cameraManipulator->getViewMatrix(), axisSize);
    }

    ImGui::End();
    ImGui::PopStyleVar();
  }
}
