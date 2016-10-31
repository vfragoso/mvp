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

// Triangle vertices (in the model space).
GLfloat vertices[] = {
  -500.0f, -500.0f, 0.f,
  500.0f, -500.0f, 0.f,
  0.0f,  500.0f, 0.f
};

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

class Model {
public:
  Model(const Eigen::Vector3f& orientation,
        const Eigen::Vector3f& position,
        const Eigen::MatrixXf& vertices) {
    orientation_ = orientation;
    position_ = position;
    vertices_ = vertices; 
  }
  ~Model() {}

  void SetOrientation(const Eigen::Vector3f& orientation) {
    orientation_ = orientation;
  }

  void SetPosition(const Eigen::Vector3f& position);

  Eigen::Vector3f* mutable_orientation() {
    return &orientation_;
  }

  Eigen::Vector3f* mutable_position() {
    return &position_;
  }

  const Eigen::Vector3f& GetOrientation() {
    return orientation_;
  }

  const Eigen::Vector3f& GetPosition() {
    return position_;
  }

private:
  // Attributes.
  Eigen::Vector3f orientation_;
  Eigen::Vector3f position_;
  Eigen::MatrixXf vertices_;
};

void Model::SetPosition(const Eigen::Vector3f& position) {
  position_ = position;
}

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
  GLfloat left, 
  GLfloat right, 
  GLfloat top, 
  GLfloat bottom, 
  GLfloat near, 
  GLfloat far) {
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

Eigen::Matrix4f ComputeProjectionMatrix(
  GLfloat field_of_view, GLfloat aspect_ratio, GLfloat near, GLfloat far) {
  GLfloat pi = 3.1416f;
  GLfloat top = near * tan((pi / 180.0f) * field_of_view * 0.5f);
  GLfloat bottom = -top;
  GLfloat right = top * aspect_ratio;
  GLfloat left = -right;
  return ComputeProjectionMatrix(left, right, top, bottom, near, far);
}



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
GLuint SetVertexBufferObject() {
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
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  // Inform OpenGL how the vertex buffer is arranged.
  constexpr GLuint kIndex = 0;  // Index of the first buffer array.
  constexpr GLuint kNumElementsInBuffer = 3;
  constexpr GLuint kBufferSize = kNumElementsInBuffer * sizeof(vertices[0]);
  const GLvoid* offset_ptr = nullptr;
  glVertexAttribPointer(kIndex, kNumElementsInBuffer, GL_FLOAT, GL_FALSE,
                        kBufferSize, offset_ptr);
  // Set as active our newly generated VBO.
  glEnableVertexAttribArray(kIndex);
  // Unbind buffer so that later we can use it.
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  return vertex_buffer_object_id;
}

// Creates and sets the vertex array object (VAO) for our triangle. Returns the
// id of the created VAO.
void SetVertexArrayObject(GLuint* vertex_buffer_object_id,
                          GLuint* vertex_array_object_id) {
  // Create the vertex array object (VAO).
  constexpr int kNumVertexArrays = 1;
  // This function creates kNumVertexArrays vaos and stores the ids in the
  // array pointed by the second argument.
  glGenVertexArrays(kNumVertexArrays, vertex_array_object_id);
  // Set the recently created vertex array object (VAO) current.
  glBindVertexArray(*vertex_array_object_id);
  // Create the Vertex Buffer Object (VBO).
  *vertex_buffer_object_id = SetVertexBufferObject();
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
    ComputeTranslation(Eigen::Vector3f(0.0f, 0.0f, sin(0.5 * angle) - 1.0f));
  Eigen::Matrix4f rotation = 
    ComputeRotation(Eigen::Vector3f::UnitZ(),
                    angle);
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
  // First argument specifies the primitive to use.
  // Second argument specifies the starting index in the VAO.
  // Third argument specified the number of vertices to use.
  glDrawArrays(GL_TRIANGLES, 0, 3);
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
  SetVertexArrayObject(&vertex_buffer_object_id, &vertex_array_object_id);

  // Create projection matrix.
  // Dimensions of image were (640, 480)
  const Eigen::Matrix4f projection_matrix = 
     ComputeProjectionMatrix(-320, 320, 240, -240, 0.1, 10);
  std::cout << projection_matrix << std::endl;
  GLfloat angle = 0.0f;  // State of rotation.

  // Loop until the user closes the window.
  while (!glfwWindowShouldClose(window)) {
    // Render the scene!
    // () casting using the C operator.
    angle = static_cast<GLfloat>(glfwGetTime()) * 10.0f;
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
