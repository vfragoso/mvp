// Copyright (C) 2016 West Virginia University.
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
//     * Neither the name of West Virginia University nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Victor Fragoso (victor.fragoso@mail.wvu.edu)

#define _USE_MATH_DEFINES  // For using M_PI.
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

// The macro below tells the linker to use the GLEW library in a static way.
// This is mainly for compatibility with Windows.
// Glew is a library that "scans" and knows what "extensions" (i.e.,
// non-standard algorithms) are available in the OpenGL implementation in the
// system. This library is crucial in determining if some features that our
// OpenGL implementation uses are not available.
#define GLEW_STATIC
#include <GL/glew.h>
// The header of GLFW. This library is a C-based and light-weight library for
// creating windows for OpenGL rendering.
// See http://www.glfw.org/ for more information.
#include <GLFW/glfw3.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "shader_program.h"

// Annonymous namespace for constants and helper functions.
namespace {
// Window dimensions.
constexpr int kWindowWidth = 640;
constexpr int kWindowHeight = 480;

// // Triangle vertices (in the model space).
// // Note that we don't use these vertices anymore, since we have now our class
// // that takes care of the vertices.
// GLfloat vertices[] = {
//   -500.0f, -500.0f, 0.f,
//   500.0f, -500.0f, 0.f,
//   0.0f,  500.0f, 0.f
// };

// GLSL shaders.
// Every shader should declare its version.
// Vertex shader follows standard 3.3.0.
// This shader declares/expexts an input variable named position. This input
// should have been loaded into GPU memory for its processing. The shader
// essentially sets the gl_Position -- an already defined variable -- that
// determines the final position for a vertex.
// Note that the position variable is of type vec3, which is a 3D dimensional
// vector. The layout keyword determines the way the VAO buffer is arranged in
// memory. This way the shader can read the vertices correctly.
const std::string vertex_shader_src =
    "#version 330 core\n"
    "layout (location = 0) in vec3 position;\n"
    "uniform mat4 model;\n"
    "uniform mat4 view;\n"
    "uniform mat4 projection;\n"
    "\n"
    "void main() {\n"
    "gl_Position = projection * view * model * vec4(position, 1.0f);\n"
    "}\n";

// Fragment shader follows standard 3.3.0. The goal of the fragment shader is to
// calculate the color of the pixel corresponding to a vertex. This is why we
// declare a variable named color of type vec4 (4D vector) as its output. This
// shader sets the output color to a (1.0, 0.5, 0.2, 1.0) using an RGBA format.
const std::string fragment_shader_src =
    "#version 330 core\n"
    "out vec4 color;\n"
    "void main() {\n"
    "color = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
    "}\n";

// Error callback function. This function follows the required signature of
// GLFW. See http://www.glfw.org/docs/3.0/group__error.html for more
// information.
static void ErrorCallback(int error, const char* description) {
  std::cerr << "ERROR: " << description << std::endl;
}

// Key callback. This function follows the required signature of GLFW. See
// http://www.glfw.org/docs/latest/input_guide.html fore more information.
static void KeyCallback(GLFWwindow* window,
                        int key,
                        int scancode,
                        int action,
                        int mods) {
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, GL_TRUE);
  }
}

// Class that will help us keep the state of any model more easily.
class Model {
public:
  // Constructor.
  // Params
  //  orientation  Axis of rotation whose norm is the angle
  //     (aka Rodrigues vector).
  //  position  The position of the object in the world.
  //  vertices  The vertices forming the object.
  Model(const Eigen::Vector3f& orientation,
        const Eigen::Vector3f& position,
        const Eigen::MatrixXf& vertices) {
    orientation_ = orientation;
    position_ = position;
    vertices_ = vertices;
  }
  // Default destructor.
  ~Model() {}

  // Setters set members by *copying* input parameters.
  void SetOrientation(const Eigen::Vector3f& orientation) {
    orientation_ = orientation;
  }

  void SetPosition(const Eigen::Vector3f& position);

  // If we want to avoid copying, we can return a pointer to
  // the member. Note that making public the attributes work
  // if we want to modify directly the members. However, this
  // is a matter of design.
  Eigen::Vector3f* mutable_orientation() {
    return &orientation_;
  }

  Eigen::Vector3f* mutable_position() {
    return &position_;
  }

  // Getters, return a const reference to the member.
  const Eigen::Vector3f& GetOrientation() {
    return orientation_;
  }

  const Eigen::Vector3f& GetPosition() {
    return position_;
  }

  const Eigen::MatrixXf& vertices() const {
    return vertices_;
  }

private:
  // Attributes.
  // The convention we will use is to define a '_' after the name
  // of the attribute.
  Eigen::Vector3f orientation_;
  Eigen::Vector3f position_;
  Eigen::MatrixXf vertices_;
};

// Implements the setter for position. Note that the class somehow defines
// a namespace.
void Model::SetPosition(const Eigen::Vector3f& position) {
  position_ = position;
}

// -------------------- Helper Functions ----------------------------------
Eigen::Matrix4f ComputeTranslation(
  const Eigen::Vector3f& offset) {
  Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
  transformation.col(3) = offset.homogeneous();
  return transformation;
}

Eigen::Matrix4f ComputeRotation(const Eigen::Vector3f& axis,
                                const GLfloat angle) {
  Eigen::AngleAxisf rotation(angle, axis);
  Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
  Eigen::Matrix3f rot3 = rotation.matrix();
  transformation.block(0, 0, 3, 3)  = rot3;
  return transformation;
}

// General form.
Eigen::Matrix4f ComputeProjectionMatrix(
  const GLfloat left, 
  const GLfloat right, 
  const GLfloat top, 
  const GLfloat bottom, 
  const GLfloat near, 
  const GLfloat far) {
  Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();
  projection(0, 0) = 2.0 * near / (right - left);
  projection(1, 1) = 2.0 * near / (top - bottom);
  projection(2, 2) = -(far + near) / (far - near);
  projection(0, 2) = (right + left) / (right - left);
  projection(1, 2) = (top + bottom) / (top - bottom);
  projection(2, 3) = -2.0 * far * near / (far - near);
  projection(3, 3) = 0.0f;
  projection(3, 2) = -1.0f;
  return projection;
}

// Mathematical constants. The right way to get PI in C++ is to use the
// macro M_PI. To do so, we have to include math.h or cmath and define
// the _USE_MATH_DEFINES macro to enable these constants. See the header
// section.
constexpr GLfloat kHalfPi = 0.5f * static_cast<GLfloat>(M_PI);

// Compute cotangent. Since C++ does not provide cotangent, we implement it
// as follows. Recall that cotangent is essentially tangent flipped and
// translated 90 degrees (or PI / 2 radians). To do the flipping and translation
// we have to do PI / 2 - angle. Subtracting the angle flips the curve.
// See the plots for http://mathworld.wolfram.com/Cotangent.html and
// http://mathworld.wolfram.com/Tangent.html
inline GLfloat ComputeCotangent(const GLfloat angle) {
  return tan(kHalfPi - angle);
}

// Reparametrization of the ComputeProjectionMatrix. This function only
// requires 4 parameters rather than 6 parameters.
Eigen::Matrix4f ComputeProjectionMatrix(const GLfloat field_of_view,
                                        const GLfloat aspect_ratio,
                                        const GLfloat near,
                                        const GLfloat far) {
  // Create the projection matrix.
  const GLfloat y_scale = ComputeCotangent(0.5f * field_of_view);
  const GLfloat x_scale = y_scale / aspect_ratio;
  const GLfloat planes_distance = far - near;
  const GLfloat z_scale =
      -(near + far) / planes_distance;
  const GLfloat homogeneous_scale =
      -2 * near * far / planes_distance;
  Eigen::Matrix4f projection_matrix;
  projection_matrix << x_scale, 0.0f, 0.0f, 0.0f,
      0.0f, y_scale, 0.0f, 0.0f,
      0.0f, 0.0f, z_scale, homogeneous_scale,
      0.0f, 0.0f, -1.0f, 0.0f;
  return projection_matrix;
}

// -------------------- End of Helper Functions --------------------------------

// Configures glfw.
void SetWindowHints() {
  // Sets properties of windows and have to be set before creation.
  // GLFW_CONTEXT_VERSION_{MAJOR|MINOR} sets the minimum OpenGL API version
  // that this program will use.
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  // Sets the OpenGL profile.
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  // Sets the property of resizability of a window.
  glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
}

// Configures the view port.
// Note: All the OpenGL functions begin with gl, and all the GLFW functions
// begin with glfw. This is because they are C-functions -- C does not have
// namespaces.
void ConfigureViewPort(GLFWwindow* window) {
  int width;
  int height;
  // We get the frame buffer dimensions and store them in width and height.
  glfwGetFramebufferSize(window, &width, &height);
  // Tells OpenGL the dimensions of the window and we specify the coordinates
  // of the lower left corner.
  glViewport(0, 0, width, height);
}

// Clears the frame buffer.
void ClearTheFrameBuffer() {
  // Sets the initial color of the framebuffer in the RGBA, R = Red, G = Green,
  // B = Blue, and A = alpha.
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  // Tells OpenGL to clear the Color buffer.
  glClear(GL_COLOR_BUFFER_BIT);
}

// Creates and transfers the vertices into the GPU. Returns the vertex buffer
// object id.
GLuint SetVertexBufferObject(const Model& model) {
  // Create a vertex buffer object (vbo).
  GLuint vertex_buffer_object_id;
  glGenBuffers(1, &vertex_buffer_object_id);
  // Set the GL_ARRAY_BUFFER of OpenGL to the vbo we just created.
  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_object_id);
  // Copy the vertices into the GL_ARRAY_BUFFER that currently 'points' to our
  // recently created vbo. In this case, sizeof(vertices) returns the size of
  // the array vertices (defined above) in bytes.
  // First parameter specifies the destination buffer.
  // Second parameter specifies the size of the buffer.
  // Third parameter specifies the pointer to the vertices buffer in RAM.
  // Fourth parameter specifies the way we want OpenGL to treat the buffer.
  // There are three different ways to treat this buffer:
  // 1. GL_STATIC_DRAW: the data will change very rarely.
  // 2. GL_DYNAMIC_DRAW: the data will likely change.
  // 3. GL_STREAM_DRAW: the data will change every time it is drawn.
  // See https://www.opengl.org/sdk/docs/man/html/glBufferData.xhtml.
  const Eigen::MatrixXf& vertices = model.vertices();
  const int vertices_size_in_bytes =
      vertices.rows() * vertices.cols() * sizeof(vertices(0, 0));
  glBufferData(GL_ARRAY_BUFFER,
               vertices_size_in_bytes,
               vertices.data(),
               GL_STATIC_DRAW);
  // Inform OpenGL how the vertex buffer is arranged.
  constexpr GLuint kIndex = 0;  // Index of the first buffer array.
  // A vertex right now contains 3 elements because we have x, y, z. But we can
  // add more information per vertex as we will see shortly.
  constexpr GLuint kNumElementsPerVertex = 3;
  constexpr GLuint kStride = kNumElementsPerVertex * sizeof(vertices(0, 0));
  const GLvoid* offset_ptr = nullptr;
  glVertexAttribPointer(kIndex, kNumElementsPerVertex, GL_FLOAT, GL_FALSE,
                        kStride, offset_ptr);
  // Set as active our newly generated VBO.
  glEnableVertexAttribArray(kIndex);
  // Unbind buffer so that later we can use it.
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  return vertex_buffer_object_id;
}

// Creates and sets the vertex array object (VAO) for our triangle. Returns the
// id of the created VAO.
void SetVertexArrayObject(const Model& model,
                          GLuint* vertex_buffer_object_id,
                          GLuint* vertex_array_object_id) {
  // Create the vertex array object (VAO).
  constexpr int kNumVertexArrays = 1;
  // This function creates kNumVertexArrays vaos and stores the ids in the
  // array pointed by the second argument.
  glGenVertexArrays(kNumVertexArrays, vertex_array_object_id);
  // Set the recently created vertex array object (VAO) current.
  glBindVertexArray(*vertex_array_object_id);
  // Create the Vertex Buffer Object (VBO).
  *vertex_buffer_object_id = SetVertexBufferObject(model);
  // Disable our created VAO.
  glBindVertexArray(0);
}

// Renders the scene.
void RenderScene(const wvu::ShaderProgram& shader_program,
                 const GLuint vertex_array_object_id,
                 const Eigen::Matrix4f& projection,
                 const GLfloat angle,
                 GLFWwindow* window) {
  // Clear the buffer.
  ClearTheFrameBuffer();
  // Let OpenGL know that we want to use our shader program.
  shader_program.Use();
  // Get the locations of the uniform variables.
  const GLint model_location = 
    glGetUniformLocation(shader_program.shader_program_id(), "model");
  const GLint view_location = 
    glGetUniformLocation(shader_program.shader_program_id(), "view");
  const GLint projection_location = 
    glGetUniformLocation(shader_program.shader_program_id(), "projection");
  Eigen::Matrix4f translation = 
    ComputeTranslation(Eigen::Vector3f(0.0f, 0.0f, -5.0f));
  Eigen::Matrix4f rotation = 
    ComputeRotation(Eigen::Vector3f::UnitZ(), angle);
  Eigen::Matrix4f model = translation * rotation;
  std::cout << "Model: \n" << model << std::endl;
  Eigen::Matrix4f view = Eigen::Matrix4f::Identity();
  // We do not create the projection matrix here because the projection 
  // matrix does not change.
  // Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();
  glUniformMatrix4fv(model_location, 1, GL_FALSE, model.data());
  glUniformMatrix4fv(view_location, 1, GL_FALSE, view.data());
  glUniformMatrix4fv(projection_location, 1, GL_FALSE, projection.data());
  // Draw the triangle.
  // Let OpenGL know what vertex array object we will use.
  glBindVertexArray(vertex_array_object_id);
  // Set to GL_LINE instead of GL_FILL to visualize the poligons as wireframes.
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  // First argument specifies the primitive to use.
  // Second argument specifies the starting index in the VAO.
  // Third argument specified the number of vertices to use.
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 8);
  // Let OpenGL know that we are done with our vertex array object.
  glBindVertexArray(0);
}

}  // namespace

int main(int argc, char** argv) {
  // Initialize the GLFW library.
  if (!glfwInit()) {
    return -1;
  }

  // Setting the error callback.
  glfwSetErrorCallback(ErrorCallback);

  // Setting Window hints.
  SetWindowHints();

  // Create a window and its OpenGL context.
  const std::string window_name = "Hello Triangle";
  GLFWwindow* window = glfwCreateWindow(kWindowWidth,
                                        kWindowHeight,
                                        window_name.c_str(),
                                        nullptr,
                                        nullptr);
  if (!window) {
    glfwTerminate();
    return -1;
  }

  // Make the window's context current.
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);
  glfwSetKeyCallback(window, KeyCallback);

  // Initialize GLEW.
  glewExperimental = GL_TRUE;
  if (glewInit() != GLEW_OK) {
    std::cerr << "Glew did not initialize properly!" << std::endl;
    glfwTerminate();
    return -1;
  }

  // Configure View Port.
  ConfigureViewPort(window);

  // Compile shaders and create shader program.
  wvu::ShaderProgram shader_program;
  shader_program.LoadVertexShaderFromString(vertex_shader_src);
  shader_program.LoadFragmentShaderFromString(fragment_shader_src);
  std::string error_info_log;
  if (!shader_program.Create(&error_info_log)) {
    std::cout << "ERROR: " << error_info_log << "\n";
  }
  // TODO(vfragoso): Implement me!
  if (!shader_program.shader_program_id()) {
    std::cerr << "ERROR: Could not create a shader program.\n";
    return -1;
  }

  // Prepare buffers to hold the vertices in GPU.
  GLuint vertex_buffer_object_id;
  GLuint vertex_array_object_id;
  Eigen::MatrixXf vertices(3, 8);
  vertices.col(0) = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
  vertices.col(1) = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
  vertices.col(2) = Eigen::Vector3f(1.0f, 1.0f, 0.0f);
  vertices.col(3) = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
  vertices.col(4) = Eigen::Vector3f(1.0f, 1.0f, -1.0f);
  vertices.col(5) = Eigen::Vector3f(1.0f, 0.0f, -1.0f);
  vertices.col(6) = Eigen::Vector3f(0.0f, 1.0f, -1.0f);
  vertices.col(7) = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
  Model model(Eigen::Vector3f(0, 0, 0),  // Orientation of object.
              Eigen::Vector3f(0, 0, 0),  // Position of object.
              vertices);
  SetVertexArrayObject(model, &vertex_buffer_object_id, &vertex_array_object_id);

  // Create projection matrix.
  const GLfloat field_of_view = 45.0f;
  const GLfloat aspect_ratio = kWindowWidth / kWindowHeight;
  const Eigen::Matrix4f projection_matrix = 
      ComputeProjectionMatrix(field_of_view, aspect_ratio, 0.1, 10);
  std::cout << projection_matrix << std::endl;
  GLfloat angle = 0.0f;  // State of rotation.

  // Loop until the user closes the window.
  const GLfloat rotation_speed = 50.0f;
  while (!glfwWindowShouldClose(window)) {
    // Render the scene!
    // Casting using (<type>) -- which is the C way -- is not recommended.
    // Instead, use static_cast<type>(input argument).
    angle = rotation_speed * static_cast<GLfloat>(glfwGetTime()) * M_PI / 180.f;
    RenderScene(shader_program, vertex_array_object_id, 
                projection_matrix, angle, window);

    // Swap front and back buffers.
    glfwSwapBuffers(window);

    // Poll for and process events.
    glfwPollEvents();
  }

  // Cleaning up tasks.
  glDeleteVertexArrays(1, &vertex_array_object_id);
  glDeleteBuffers(1, &vertex_buffer_object_id);
  // Destroy window.
  glfwDestroyWindow(window);
  // Tear down GLFW library.
  glfwTerminate();

  return 0;
}
