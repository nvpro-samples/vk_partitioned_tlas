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
(Totally inaccurate) Physics simulation for dominoes and update of their partition assignments
*/

#version 460
#extension GL_GOOGLE_include_directive : enable
#include "animation_common.h"


// Constants
const vec3  gravity                   = vec3(0.0, -9.81, 0.0);
const float mass                      = 1.0;        // Assume unit mass for simplicity
const float restitution               = 0.1;        // Coefficient of restitution for collisions
const float groundFrictionCoefficient = 1.;         // Friction coefficient
const float dominoFrictionCoefficient = 0.2;        // Friction coefficient
const mat3  inertiaTensorInv          = mat3(1.0);  // Inverse inertia tensor (assuming a unit cube for simplicity)

float dampingFactor = 1.f - constants.timeStep;


vec3 getPartitionPosition(uint32_t partitionId)
{
  uint32_t x = partitionId % constants.partitionsPerAxis;
  uint32_t y = partitionId / constants.partitionsPerAxis;

  vec3 bbSize = constants.objectBboxMax - constants.objectBboxMin;

  float dx = (0.5f + float(x)) * (bbSize.x / float(constants.partitionsPerAxis));
  float dy = (0.5f + float(y)) * (bbSize.z / float(constants.partitionsPerAxis));

  return constants.objectBboxMin + vec3(dx, 0, dy);
}

mat4 makeRotation(float angle, vec3 axis)
{
  axis     = normalize(axis);
  float s  = sin(angle);
  float c  = cos(angle);
  float oc = 1.0 - c;

  return mat4(oc * axis.x * axis.x + c, oc * axis.x * axis.y - axis.z * s, oc * axis.z * axis.x + axis.y * s, 0.0,
              oc * axis.x * axis.y + axis.z * s, oc * axis.y * axis.y + c, oc * axis.y * axis.z - axis.x * s, 0.0,
              oc * axis.z * axis.x - axis.y * s, oc * axis.y * axis.z + axis.x * s, oc * axis.z * axis.z + c, 0.0, 0.0,
              0.0, 0.0, 1.0);
}


mat3x4 makeRotation3x4(float angle, vec3 axis)
{
  axis     = normalize(axis);
  float s  = sin(angle);
  float c  = cos(angle);
  float oc = 1.0 - c;

  return mat3x4(oc * axis.x * axis.x + c, oc * axis.x * axis.y + axis.z * s, oc * axis.z * axis.x - axis.y * s, 0.0,
                oc * axis.x * axis.y - axis.z * s, oc * axis.y * axis.y + c, oc * axis.y * axis.z + axis.x * s, 0.0,
                oc * axis.z * axis.x + axis.y * s, oc * axis.y * axis.z - axis.x * s, oc * axis.z * axis.z + c, 0.0);
}


float getGroundHeight(vec2 location)
{
  return 0.f;
}

const vec3 cubeCorners[8] = vec3[8](vec3(-0.5, -0.5, -0.5),
                                    vec3(0.5, -0.5, -0.5),
                                    vec3(-0.5, 0.5, -0.5),
                                    vec3(0.5, 0.5, -0.5),
                                    vec3(-0.5, -0.5, 0.5),
                                    vec3(0.5, -0.5, 0.5),
                                    vec3(-0.5, 0.5, 0.5),
                                    vec3(0.5, 0.5, 0.5));


vec3[8] getTransformedCorners(AnimationState state)
{
  vec3 transformedCorners[8];

  [[unroll]] for(int i = 0; i < 8; i++)
  {
    transformedCorners[i] = transformPoint(state, cubeCorners[i]);
  }
  return transformedCorners;
}


void handleCollisionsAndUpdate(inout AnimationState state, vec3 transformedCorners[8], float deltaTime)
{
  bool  collisionDetected = false;
  vec3  collisionPoint    = vec3(0);
  vec3  collisionNormal   = vec3(0.0, 1.0, 0.0);
  float minY              = 1e34f;
  uint  collisionCount    = 0;

  float groundTolerance    = 0.01f;
  uint  nearCollisionCount = 0;

  for(int i = 0; i < 8; i++)
  {
    float ground = getGroundHeight(transformedCorners[i].xz);
    if(transformedCorners[i].y <= ground)
    {
      collisionDetected = true;
      collisionCount++;
      minY = min(minY, transformedCorners[i].y);
      collisionPoint += transformedCorners[i] - ground;
    }
    if(abs(transformedCorners[i].y - ground) < groundTolerance)
    {
      nearCollisionCount++;
    }
  }
  bool isCloseToGround = (nearCollisionCount > 2) && state.lastContact != ~0u && (constants.frameIndex - state.lastContact) > 50;
  if(isCloseToGround)
  {
    addFlag(state.stateID, STATE_CLOSE_TO_GROUND);
    state.angularVelocity  = vec3(0);
    state.linearVelocity.y = 0;
    state.linearVelocity.xz *= 0.1f;
    return;
  }

  if(collisionDetected)
  {

    setCollisionState(state.stateID, STATE_GROUND_COLLISION);

    collisionPoint /= collisionCount;
    mat3 scaleMatrix            = getTransform3x3(state);
    mat3 scaledInertiaTensorInv = inverse(transpose(scaleMatrix) * inertiaTensorInv * scaleMatrix);

    vec3 r                = collisionPoint - vec3(getPosition(state));
    vec3 velocityAtPoint  = state.linearVelocity - cross(state.angularVelocity, r);
    vec3 relativeVelocity = velocityAtPoint;

    if(length(relativeVelocity) < 0.1f)
    {
      relativeVelocity = vec3(0.f);
    }

    float normalImpulseMagnitude =
        -(1.0 + restitution) * dot(relativeVelocity, collisionNormal)
        / (1.0 / mass + dot(cross(scaledInertiaTensorInv * cross(r, collisionNormal), r), collisionNormal));
    vec3 impulse = normalImpulseMagnitude * collisionNormal;
    state.linearVelocity += impulse / mass;

    vec3 tangentVelocity = relativeVelocity - dot(relativeVelocity, collisionNormal) * collisionNormal;
    if(length(tangentVelocity) > 0.0)
    {
      vec3 tangentImpulse = -normalize(tangentVelocity) * groundFrictionCoefficient * length(tangentVelocity);
      state.linearVelocity += tangentImpulse / mass;
    }

    vec3 gravityTorque = cross(r, gravity);
    if(length(gravityTorque) < 0.1f)
    {
      gravityTorque = vec3(0.f);
    }

    float dampingVelocity = 1.f;
    if(!isCloseToGround)
    {
      state.angularVelocity += scaledInertiaTensorInv * gravityTorque * deltaTime;
    }

    setPosition(state, getPosition(state) + vec3(0.0, max(0.f, getGroundHeight(getPosition(state).xz) - minY), 0.0));
  }
  else
  {
    state.linearVelocity += gravity * deltaTime;
  }

  float lowVelocityThreshold = 0.01f;
  float ground               = getGroundHeight(getPosition(state).xz);
  float positionThreshold    = ground + 0.1f;

  if(getPositionY(state) < positionThreshold && length(state.linearVelocity) < lowVelocityThreshold
     && length(state.angularVelocity) < lowVelocityThreshold)
  {
    state.linearVelocity  = vec3(0);
    state.angularVelocity = vec3(0);
    setPositionY(state, max(getPositionY(state), positionThreshold));  // Ensure it stays at ground level
  }
}


void updateAngularMotion(inout AnimationState state, float deltaTime)
{
  float angle = length(state.angularVelocity) * deltaTime;
  if(angle > 0.0)
  {
    vec3   axis           = normalize(state.angularVelocity);
    mat3x4 rotationMatrix = makeRotation3x4(angle, axis);

    vec3 p = getPosition(state);
    setPosition(state, vec3(0));
    mulTransform3x4(state, rotationMatrix);
    setPosition(state, p);
  }
}

AnimationState simulateDominoPhysics(AnimationState state, vec3 corners[8], float deltaTime)
{

  AnimationState s = state;
  handleCollisionsAndUpdate(s, corners, deltaTime);
  return s;
}


struct Collision
{
  vec3  point;
  vec3  normal;
  float penetrationDepth;
};


bool detectCollision(AnimationState state1, vec3 corners1[8], AnimationState state2, vec3 corners2[8], out Collision collision)
{

  vec3 axes[15];
  axes[0] = normalize(corners1[1] - corners1[0]);  // X-axis of box1
  axes[1] = normalize(corners1[3] - corners1[0]);  // Y-axis of box1
  axes[2] = normalize(corners1[4] - corners1[0]);  // Z-axis of box1
  axes[3] = normalize(corners2[1] - corners2[0]);  // X-axis of box2
  axes[4] = normalize(corners2[3] - corners2[0]);  // Y-axis of box2
  axes[5] = normalize(corners2[4] - corners2[0]);  // Z-axis of box2

  int index = 6;
  for(int i = 0; i < 3; ++i)
  {
    for(int j = 3; j < 6; ++j)
    {
      vec3 axis = cross(axes[i], axes[j]);
      if(length(axis) < 1e-6)
        continue;  // Skip near-zero-length axes
      axes[index++] = normalize(axis);
    }
  }

  float minPenetration = 1e34;
  vec3  minPenetrationAxis;
  vec3  minContactPoint1;
  vec3  minContactPoint2;

  for(int i = 0; i < index; ++i)
  {
    vec3 axis = axes[i];

    float minProj1 = dot(corners1[0], axis), maxProj1 = minProj1;
    float minProj2 = dot(corners2[0], axis), maxProj2 = minProj2;

    for(int k = 1; k < 8; ++k)
    {
      float proj1 = dot(corners1[k], axis);
      minProj1    = min(minProj1, proj1);
      maxProj1    = max(maxProj1, proj1);

      float proj2 = dot(corners2[k], axis);
      minProj2    = min(minProj2, proj2);
      maxProj2    = max(maxProj2, proj2);
    }

    float penetration = min(maxProj1, maxProj2) - max(minProj1, minProj2);
    if(penetration < 0)
    {
      return false;
    }

    if(penetration < minPenetration)
    {
      minPenetration     = penetration;
      minPenetrationAxis = axis;

      // Find the support points on the surfaces of both boxes
      vec3 supportPoint1 = corners1[0];
      vec3 supportPoint2 = corners2[0];
      for(int j = 1; j < 8; ++j)
      {
        if(dot(corners1[j], axis) < dot(supportPoint1, axis))
        {
          supportPoint1 = corners1[j];
        }
        if(dot(corners2[j], axis) > dot(supportPoint2, axis))
        {
          supportPoint2 = corners2[j];
        }
      }

      minContactPoint1 = supportPoint1;
      minContactPoint2 = supportPoint2;
    }
  }

  // Adjust contact points to lie on the surfaces
  vec3 contactPoint1 =
      minContactPoint1 - minPenetrationAxis * (dot(minContactPoint1 - minContactPoint2, minPenetrationAxis) * 0.5);
  vec3 contactPoint2 =
      minContactPoint2 + minPenetrationAxis * (dot(minContactPoint1 - minContactPoint2, minPenetrationAxis) * 0.5);

  collision.point            = (contactPoint1 + contactPoint2) * 0.5;
  collision.normal           = minPenetrationAxis;
  collision.penetrationDepth = minPenetration;
  return true;
}


vec3 lowestCorner(vec3 corners[8])
{
  vec3 selected = corners[0];
  for(uint i = 1; i < 8; i++)
  {
    if(corners[i].y < selected.y)
    {
      selected = corners[i];
    }
  }
  return selected;
}

uint pcg(inout uint state)
{
  uint prev = state * 747796405u + 2891336453u;
  uint word = ((prev >> ((prev >> 28u) + 4u)) ^ prev) * 277803737u;
  state     = prev;
  return (word >> 22u) ^ word;
}
uint32_t wangHash(uint32_t seed)
{
  seed = (seed ^ 61) ^ (seed >> 16);
  seed *= 9;
  seed = seed ^ (seed >> 4);
  seed *= 0x27d4eb2d;
  seed = seed ^ (seed >> 15);
  return seed;
}


bool handleInteraction(inout AnimationState prevDomino, vec3 prevCorners[8], inout AnimationState currDomino, vec3 currCorners[8], float deltaTime)
{
  Collision collision;


  if(currDomino.firstContact != ~0u && constants.frameIndex - currDomino.firstContact > 300)
  {
    return false;
  }

  float transferRatio    = 1.0f;
  float localRestitution = 0.1f;
  float mass             = 1.0f;  // Example mass, adjust as needed

  if(detectCollision(prevDomino, prevCorners, currDomino, currCorners, collision))
  {
    if(currDomino.firstContact == ~0u)
    {
      currDomino.firstContact = constants.frameIndex;
    }
    setCollisionState(currDomino.stateID, STATE_OTHER_COLLISION);
    setCollisionState(prevDomino.stateID, STATE_OTHER_COLLISION);

    vec3 bottom1 = lowestCorner(currCorners);
    vec3 bottom2 = lowestCorner(prevCorners);


    if(hasFlag(currDomino.stateID, STATE_CLOSE_TO_GROUND))
    {
      setPosition(currDomino, getPosition(currDomino) - collision.normal * collision.penetrationDepth * 0.5f);
      currDomino.linearVelocity -= collision.normal * collision.penetrationDepth * 0.5f;
    }

    const float maxRotation = 1.f;

    const float rotationMultiplier = 5.f;

    // FIXME!! Use proper inertia tensor for this
    {
      vec3 ground = vec3(0.f, -0.5f, 0.f);
      ground      = transformPoint(currDomino, ground);

      float dist = length(collision.point - ground);
      if(dist != 0.f)
      {


        vec3 rotation = -rotationMultiplier * cross(collision.normal, vec3(0, 1, 0)) * collision.penetrationDepth / dist;
        rotation = clamp(rotation, vec3(-maxRotation), vec3(maxRotation));
        if(length(currDomino.angularVelocity) < maxRotation)
        {
          currDomino.angularVelocity += rotation;
        }
      }
    }
    vec3 direction = normalize(getPosition(currDomino) - getPosition(prevDomino));

    float correctionSign = dot(direction, collision.normal) < 0.f ? 1.f : -1.f;

    bottom1 += correctionSign * collision.normal * collision.penetrationDepth * 0.5f;

    {
      float ground = getGroundHeight(bottom1.xz);
      if(bottom1.y < ground)
      {
        setPositionY(currDomino, getPositionY(currDomino) + ground - bottom1.y);
        currDomino.linearVelocity.y = 0.f;
      }
    }
    setPosition(prevDomino, getPosition(prevDomino) + collision.normal * collision.penetrationDepth * 0.5f);
    prevDomino.linearVelocity += collision.normal * collision.penetrationDepth * 0.5f;
    // FIXME!! Use proper inertia tensor for this
    {
      vec3 ground = vec3(0.f, -0.5f, 0.f);
      ground      = transformPoint(prevDomino, ground);


      float dist = length(collision.point - ground);
      if(dist != 0.f)
      {
        vec3 rotation = rotationMultiplier * cross(collision.normal, vec3(0, 1, 0)) * collision.penetrationDepth / dist;
        rotation      = clamp(rotation, vec3(-maxRotation), vec3(maxRotation));

        if(length(prevDomino.angularVelocity) < maxRotation)
        {
          prevDomino.angularVelocity += rotation;
        }
      }
    }
    bottom2 += -correctionSign * collision.normal * collision.penetrationDepth * 0.5f;
    {
      float ground = getGroundHeight(getPosition(prevDomino).xz);
      if(bottom2.y < ground)
      {
        setPositionY(prevDomino, getPositionY(prevDomino) + ground - bottom2.y);
        prevDomino.linearVelocity.y = 0.f;
      }
    }


    vec3 rPrev = collision.point - getPosition(prevDomino);
    vec3 rCurr = collision.point - getPosition(currDomino);

    vec3 velocityAtPointPrev = prevDomino.linearVelocity + cross(prevDomino.angularVelocity, rPrev);
    vec3 velocityAtPointCurr = currDomino.linearVelocity + cross(currDomino.angularVelocity, rCurr);

    vec3  relativeVelocity       = velocityAtPointCurr - velocityAtPointPrev;
    float relativeNormalVelocity = dot(relativeVelocity, collision.normal);

    if(relativeNormalVelocity > 0)
      return false;  // Dominos are moving apart, no impulse needed

    //float normalImpulseMagnitude = -(1.0 + localRestitution) * relativeNormalVelocity /
    //  (1.0 / mass + 1.0 / mass +
    //    (dot(cross(inverse(mat3(currDomino.transform)) * cross(rCurr, collision.normal), rCurr), collision.normal)) / mass +
    //    (dot(cross(inverse(mat3(prevDomino.transform)) * cross(rPrev, collision.normal), rPrev), collision.normal)) / mass);

    float normalImpulseMagnitude =
        -(1.0 + localRestitution) * relativeNormalVelocity
        / (2.0 / mass
           + abs((dot(cross(inverse((getTransform3x3(currDomino))) * cross(rCurr, collision.normal), rCurr), collision.normal)) / mass
                 - (dot(cross(inverse((getTransform3x3(prevDomino))) * cross(rPrev, collision.normal), rPrev), collision.normal)) / mass));


    normalImpulseMagnitude *= dominoFrictionCoefficient;


    vec3 impulse = normalImpulseMagnitude * collision.normal;

    // Apply impulse to linear velocities
    currDomino.linearVelocity += impulse / mass;
    prevDomino.linearVelocity -= impulse / mass;

    // Apply impulse to angular velocities

    //mat3 currScaleMatrix = mat3(currDomino.transform);
    //mat3 currScaledInertiaTensorInv = inverse(transpose(currScaleMatrix) * inertiaTensorInv * currScaleMatrix);
    //
    //mat3 prevScaleMatrix = mat3(prevDomino.transform);
    //mat3 prevScaledInertiaTensorInv = inverse(transpose(prevScaleMatrix) * inertiaTensorInv * prevScaleMatrix);
    //
    //currDomino.angularVelocity += currScaledInertiaTensorInv * cross(rCurr, impulse);// / mass;
    //prevDomino.angularVelocity -= prevScaledInertiaTensorInv * cross(rPrev, impulse);// / mass;


    currDomino.angularVelocity -= inverse(getTransform3x3(currDomino)) * cross(rCurr, impulse) / mass;
    prevDomino.angularVelocity += inverse(getTransform3x3(prevDomino)) * cross(rPrev, impulse) / mass;

    // Apply friction impulse
    vec3 tangentVelocity = relativeVelocity - relativeNormalVelocity * collision.normal;
    if(length(tangentVelocity) > 0.0)
    {
      vec3 tangentDirection = normalize(tangentVelocity);
      //float frictionImpulseMagnitude = length(tangentVelocity) * frictionCoefficient;

      vec3 frictionImpulse =
          -(1.0 + localRestitution) * tangentVelocity
          / (2.0 / mass
             + abs((dot(cross(inverse(getTransform3x3(currDomino)) * cross(rCurr, collision.normal), rCurr), collision.normal)) / mass
                   - (dot(cross(inverse(getTransform3x3(prevDomino)) * cross(rPrev, collision.normal), rPrev), collision.normal)) / mass));
      frictionImpulse *= dominoFrictionCoefficient;


      //vec3 frictionImpulse = -frictionImpulseMagnitude * tangentDirection;

      // Apply friction impulse to linear velocities
      //currDomino.linearVelocity += frictionImpulse / mass;
      //prevDomino.linearVelocity -= frictionImpulse / mass;

      // Apply friction impulse to angular velocities
      //currDomino.angularVelocity -= inverse(mat3(currDomino.transform)) * cross(rCurr, frictionImpulse) / mass;
      //prevDomino.angularVelocity += inverse(mat3(prevDomino.transform)) * cross(rPrev, frictionImpulse) / mass;
    }

    uint seed = wangHash(floatBitsToUint(getPosition(currDomino).x));
    for(uint32_t i = 0; i < 3; i++)
    {
      float rndValue = ((2.f * float(pcg(seed)) / float(~0u)) - 1.f) * 0.01f;
      currDomino.linearVelocity[i] *= rndValue + 1.f;
      rndValue = ((2.f * float(pcg(seed)) / float(~0u)) - 1.f) * 0.01f;
      currDomino.angularVelocity[i] *= rndValue + 1.f;
    }

    return true;
  }
  return false;
}


bool computeDomino(int index, inout AnimationState currDomino, inout AnimationState prevDomino, inout AnimationState nextDomino)
{
  setCollisionState(currDomino.stateID, STATE_FREE);
  mat3x4 originalTransform = getTransform3x4(currDomino);
  bool   isForced          = false;
  if(hasFlag(currDomino.stateID, STATE_FORCE_TOPPLE))
  {
    vec3 direction = normalize(transformPoint(currDomino, vec3(0.f, 0.f, 1.f)) - getPosition(currDomino));
    currDomino.angularVelocity += direction * 5.f;
    removeFlag(currDomino.stateID, STATE_FORCE_TOPPLE);
    isForced = true;
  }

  vec3 currCorners[8] = getTransformedCorners(currDomino);


  currDomino = simulateDominoPhysics(currDomino, currCorners, constants.timeStep);


  if(index > 0)
  {
    vec3 prevCorners[8] = getTransformedCorners(prevDomino);
    if(handleInteraction(prevDomino, prevCorners, currDomino, currCorners, constants.timeStep) && index < 1000)
    {
    }
  }

  if(index < constants.dynamicObjectCount - 1)
  {
    vec3 nextCorners[8] = getTransformedCorners(nextDomino);
    if(handleInteraction(currDomino, currCorners, nextDomino, nextCorners, constants.timeStep))
    {
    }
  }


  vec3 position = getPosition(currDomino);
  position += currDomino.linearVelocity * constants.timeStep;
  setPosition(currDomino, position);

  vec3  cornersBeforeRotation[8] = getTransformedCorners(currDomino);
  float minYBefore               = cornersBeforeRotation[0].y;
  for(uint i = 1; i < 8; i++)
  {
    minYBefore = min(minYBefore, cornersBeforeRotation[i].y);
  }

  updateAngularMotion(currDomino, constants.timeStep);

  vec3  cornersAfterRotation[8] = getTransformedCorners(currDomino);
  float minYAfter               = cornersAfterRotation[0].y;
  for(uint i = 1; i < 8; i++)
  {
    minYAfter = min(minYAfter, cornersAfterRotation[i].y);
  }

  float ground = getGroundHeight(getPosition(currDomino).xz);

  if(minYAfter < ground && minYBefore >= ground)
  {
    currDomino.linearVelocity.y = max(currDomino.linearVelocity.y, 0.f);
    setPositionY(currDomino, getPositionY(currDomino) + ground - minYAfter);
  }

  uint32_t contactPoints = 0;
  for(uint i = 1; i < 8; i++)
  {
    if(cornersBeforeRotation[i].y < ground + 0.01f)
    {
      contactPoints++;
    }
  }

  if(hasFlag(currDomino.stateID, STATE_OTHER_COLLISION) || isForced)
  {
    currDomino.lastContact = constants.frameIndex;
  }


  bool animationFinished = (currDomino.lastContact != ~0u) && (constants.frameIndex - currDomino.lastContact > 300);

  if(animationFinished)
  {
    currDomino.angularVelocity *= 0.01f;
    currDomino.linearVelocity *= 0.01f;

    if(length(currDomino.angularVelocity) < 0.02f && length(currDomino.linearVelocity) < 0.02f)
    {
      currDomino.angularVelocity = vec3(0.f);
      currDomino.linearVelocity  = vec3(0.f);

      setTransform3x4(currDomino, originalTransform);
    }
  }

  bool isColliding = hasFlag(currDomino.stateID, STATE_GROUND_COLLISION) || hasFlag(currDomino.stateID, STATE_OTHER_COLLISION);

  if(isColliding || currDomino.linearVelocity.y > 0.f)
  {
    currDomino.linearVelocity *= dampingFactor;
  }
  currDomino.angularVelocity *= dampingFactor;


  if(getTransform3x4(currDomino) != originalTransform)
  {
    if(currDomino.firstContact == constants.frameIndex)
    {
      atomicMax(AnimationGlobalStateBuffer(constants.globalState).s.currentCollisionIndex, index);
    }

    return true;
  }


  return false;
}


void main()
{
  // Index of the domino within the list of dominoes
  uint32_t dynamicIndex = (gl_GlobalInvocationID.x < constants.dynamicObjectCount) ? gl_GlobalInvocationID.x : ~0u;
  // Index of the domino within the global object list
  uint32_t globalIndex = dynamicIndexToGlobalIndex(dynamicIndex, constants);

  if(dynamicIndex == ~0u)
  {
    return;
  }

  // Load the current domino as well as its two neighbors, if they exist
  AnimationState currDomino = AnimationStateBuffer(constants.state[(constants.currentStateIndex) % 2]).s[dynamicIndex];
  AnimationState prevDomino, nextDomino;
  if(dynamicIndex > 0)
  {
    prevDomino = AnimationStateBuffer(constants.state[(constants.currentStateIndex) % 2]).s[dynamicIndex - 1];
  }
  if(dynamicIndex < constants.dynamicObjectCount - 1)
  {
    nextDomino = AnimationStateBuffer(constants.state[(constants.currentStateIndex) % 2]).s[dynamicIndex + 1];
  }


  bool isUpdated = false;
  // Handle physics for the current domino
  for(uint32_t i = 0; i < constants.subframeCount; i++)
  {
    bool updated = computeDomino(int(dynamicIndex), currDomino, prevDomino, nextDomino);
    isUpdated    = isUpdated || updated;
  }

  // If the domino has been updated, update the (P)TLAS instance data
  if(isUpdated)
  {
    // Update the counter for instance updates, used for UI only
    if(currDomino.lastModified != constants.frameIndex)
    {
      atomicAdd(AnimationGlobalStateBuffer(constants.globalState).s.instanceUpdateCount, 1);
    }
    currDomino.lastModified = constants.frameIndex;

    // Mark the partition as containing a modified domino. Combined with the chosen dynamic rebuild behavior, will trigger
    // a parttion rebuild or a move of the domino to the global partition
    uint32_t lastModified =
        atomicExchange(PartitionStateBuffer(constants.partitionState).s[currDomino.partitionID].lastModified, constants.frameIndex);

    // Partitioned TLAS path, handling the move of dominoes between partitions if needed
    if(constants.ptlasActive == 1)
    {
      // If the domino has already been moved into the global partition, mark its original partition as containing a modified domino
      // This will be used in combination with constants.dynamicMarkAllDominos == true and
      // constants.dynamicUpdateMode == PTLAS_DYNAMIC_UPDATE_OR_MOVE_TO_GLOBAL || constants.dynamicUpdateMode == PTLAS_DYNAMIC_MOVE_TO_GLOBAL
      // so the dominoes that belonged to a given partition are moved back from global to their original partition when none of them are moving anymore
      if(currDomino.partitionID == constants.globalPartitionIndex)
      {
        uint32_t originalPartitionID = AnimationStateBuffer(constants.originalState).s[dynamicIndex].partitionID;
        atomicExchange(PartitionStateBuffer(constants.partitionState).s[originalPartitionID].lastModified, constants.frameIndex);
      }

      // Check whether the domino should be moved to the global partition
      bool canMoveToGlobal = (constants.dynamicUpdateMode == PTLAS_DYNAMIC_MOVE_TO_GLOBAL);
      if(constants.dynamicUpdateMode == PTLAS_DYNAMIC_UPDATE_OR_MOVE_TO_GLOBAL && currDomino.partitionID != constants.globalPartitionIndex)
      {
        // If the decision to move to global partition depends on the distance to the viewpoint, compute the distance to the center of the
        // partition the domino belongs to and compare it to the user-defined threshold
        vec3  partitionPos = getPartitionPosition(currDomino.partitionID);
        vec3  toCam        = partitionPos - constants.eyePosition;
        float sqDist       = dot(toCam, toCam);
        if(sqDist >= constants.dynamicDistanceThreshold * constants.dynamicDistanceThreshold)
        {
          canMoveToGlobal = true;
        }
      }

      // Move the domino to the global partition if the user parameters allow and the domino is not already there
      bool needMoveToGlobal = canMoveToGlobal && (currDomino.partitionID != constants.globalPartitionIndex);
      if(needMoveToGlobal)
      {
        // Mark the domino for move to global partition
        currDomino.newPartitionID = constants.globalPartitionIndex;
        // The partitions involved in the move (the original and the global) will need to have their instance indices rewritten.
        // Reset the counter so the instance definitions are rewritten in the instance update kernel
        PartitionStateBuffer(constants.partitionState).s[currDomino.partitionID].dynamicWriteSlot    = 0;
        PartitionStateBuffer(constants.partitionState).s[currDomino.newPartitionID].dynamicWriteSlot = 0;
        // The per-partition instance lists must be compact arrays. To add/remove a domino from a partition we then
        // rewrite its entire instance index list in the instnace update kernel
        PartitionStateBuffer(constants.partitionState).s[currDomino.newPartitionID].needInstanceIndicesRewrite = 1;
        PartitionStateBuffer(constants.partitionState).s[currDomino.partitionID].needInstanceIndicesRewrite    = 1;
      }
    }
    else
    {
      // Regular TLAS path, update the instance data directly
      AccelerationStructureInstanceBuffer(constants.instancesAddress).i[globalIndex].transform = getTransform3x4(currDomino);
    }
    // Write the updated domino back to the animation state buffer
    AnimationStateBuffer(constants.state[(constants.currentStateIndex + 1) % 2]).s[dynamicIndex] = currDomino;
  }
  else
  {
    // If the domino remained static we just need to check if it needs to be brought back from the global partition to its original partition
    if(constants.ptlasActive == 1)
    {
      // If this is the first frame the domino is static, copy its state to the next frame. Do this for two frames so the
      // velocities both states are identical
      if(currDomino.lastModified >= constants.frameIndex - 2)
      {
        AnimationStateBuffer(constants.state[(constants.currentStateIndex + 1) % 2]).s[dynamicIndex] = currDomino;
      }

      // The domino may be moved only if it is in the global partition
      bool needMoveFromGlobal = (currDomino.partitionID == constants.globalPartitionIndex);

      if(needMoveFromGlobal)
      {
        // If dominoes are moved to the global partition on a per-partition basis (rather than individually), check whether
        // any domino in the original partition has been modified in the last frame. If so, keep the domino in the global partition
        if(constants.dynamicMarkAllDominos != 0)
        {
          uint32_t originalPartitionID = AnimationStateBuffer(constants.originalState).s[dynamicIndex].partitionID;
          if(PartitionStateBuffer(constants.partitionState).s[originalPartitionID].lastModified >= constants.frameIndex - 1)
          {
            needMoveFromGlobal = false;
          }
        }
      }
      // If needed, mark the domino for move to its original partition
      if(needMoveFromGlobal)
      {
        currDomino.newPartitionID = AnimationStateBuffer(constants.originalState).s[dynamicIndex].partitionID;
        AnimationStateBuffer(constants.state[(constants.currentStateIndex + 1) % 2]).s[dynamicIndex].newPartitionID =
            currDomino.newPartitionID;

        // The per-partition instance lists must be compact arrays. To add/remove a domino from a partition we then
        // rewrite its entire instance index list in the instance update kernel
        PartitionStateBuffer(constants.partitionState).s[currDomino.newPartitionID].needInstanceIndicesRewrite = 1;
        PartitionStateBuffer(constants.partitionState).s[currDomino.partitionID].needInstanceIndicesRewrite    = 1;
      }
      else
      {
        AnimationStateBuffer(constants.state[(constants.currentStateIndex + 1) % 2]).s[dynamicIndex].newPartitionID =
            currDomino.partitionID;
      }
    }
  }
}