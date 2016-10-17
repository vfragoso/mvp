// Copyright (C) 2016  Victor Fragoso <victor.fragoso@mail.wvu.edu>
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of the West Virginia University nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL VICTOR FRAGOSO BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include "assignment.h"

#include <math.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace wvu {
// Adds two 3d points and returns the resultant added point.
Eigen::Vector3f Add3dPoints(const Eigen::Vector3f& x,
                            const Eigen::Vector3f& y) {
  return x + y;
}

// Adds two 4d points and returns the resultant added point.
Eigen::Vector4f Add4dPoints(const Eigen::Vector4f& x,
                            const Eigen::Vector4f& y) {
  return x + y;
}

// Multiply two 4x4 matrices.
Eigen::Matrix4f Multiply4x4Matrices(const Eigen::Matrix4f& x,
                                    const Eigen::Matrix4f& y) {
  return x * y;
}

// Multiply matrix-vector. Returns the multiplication.
Eigen::Vector4f MultiplyVectorAndMatrix(const Eigen::Matrix4f& x,
                                        const Eigen::Vector4f& y) {
  return x * y;
}

// Calculate the dot product of two vectors.
float ComputeDotProduct(const Eigen::Vector3f& x,
                        const Eigen::Vector3f& y) {
  return x.dot(y);
}

// Calculates the angle between two vectors in radians.
float CalculateAngleBetweenTwoVectors(const Eigen::Vector3f& x,
                                      const Eigen::Vector3f& y) {
  const float cos_theta = x.normalized().dot(y.normalized());
  return acos(cos_theta);
}

// Calculates the cross product of two vectors.
Eigen::Vector3f ComputeCrossProduct(const Eigen::Vector3f& x,
                                    const Eigen::Vector3f& y) {
  return x.cross(y);
}

}  // namespace
