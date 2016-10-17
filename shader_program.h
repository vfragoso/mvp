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

#ifndef GLUTILS_SHADER_PROGRAM_H_
#define GLUTILS_SHADER_PROGRAM_H_

#include <string>
#include <GL/glew.h>

namespace wvu {
// This class helps with the compilation of vertex and fragment shaders. The
// class compiles the shaders and creates a shader program. The class keeps
// the id of such a compiled and linked program. The class also provides a way
// to use the shader by calling the Use() member function.
// The class can load shaders from file or accept C++ strings holding the
// contents of the shader.
// To use the class simply create an instance, load shaders from string or files
// and then call the Create() function.
// When the user desires to use the shader, the member function Use() should be
// called prior rendering.
//
// Examples.
// 1) Loading shaders from files example:
//
// wvu::ShaderProgram shader_program;
// shader_program.LoadVertexShaderFromFile("/absolute/path/to/vertex_shader");
// shader_program.LoadVertexShaderFromFile("/absolute/path/to/fragment_shader");
// std::string error_info_log;
// if (!shader_program.Create(&error_info_log)) {
//   std::cerr << error_info_log << std::endl;
// }
//
// 2) Loading shaders from C++ strings example:
//
// wvu::ShaderProgram shader_program;
// shader_program.LoadVertexShaderFromString(vertex_shader_string_instance);
// shader_program.LoadVertexShaderFromString(fragment_shader_string_instance);
// std::string error_info_log;
// if (!shader_program.Create(&error_info_log)) {
//   LOG(ERROR) << error_info_log;
// }
//
// 3) Using the shader program example:
//
// while (...) {  // Rendering loop.
//   ...
//   shader_program.Use();
//   shader_program.PassUniformVariables(drawable_object, camera);
//   ...
// }
//
// 4) Passing uniform variables to shader example:
// When passing values to uniform variables in the shader program, the shader
// program id is necessary. This class provides access to this id by calling the
// accessor method shader_program_id().
//
//  ...
//  const GLint uniform_variable_location =
//     glGetUniformLocation(shader_program_.shader_program_id(),
//                          "uniform_variable_name");
//  ...
class ShaderProgram {
 public:
  // Default constructor.
  ShaderProgram() :
      // Initializing member attributes.
      vertex_shader_src_(""), fragment_shader_src_(""),
      vertex_shader_(0), fragment_shader_(0), shader_program_id_(0),
      created_(false) {}
  // Destructor. Invoked automatically once the instance goes out of scope.
  virtual ~ShaderProgram() {
    if (created_) {
      // Once the shader program is not needed, we tell OpenGL to delete it.
      glDeleteProgram(shader_program_id_);
    }
  }

  // The accessor member returns the shader program id that OpenGL generates
  // when creating the shader program. When the shader program has not been
  // created, the shader_program_id() returns 0.
  GLuint shader_program_id() const {
    return shader_program_id_;
  }

  // Loads a vertex shader source coude from a string. Returns true if
  // successful, and false otherwise.
  // Parameters:
  //   vertex_shader_source  The C++ string containing the vertex shader source.
  bool LoadVertexShaderFromString(const std::string& vertex_shader_source);

  // Loads a fragment shader source coude from a string. Returns true if
  // successful, and false otherwise.
  // Parameters:
  //   fragment_shader_source  The C++ string containing the fragment shader
  //     source.
  bool LoadFragmentShaderFromString(const std::string& fragment_shader_source);

  // Loads a vertex shader from a file. Returns true if
  // successful, and false otherwise.
  // Parameters:
  //   vertex_shader_path  The filepath for the vertex shader.
  bool LoadVertexShaderFromFile(const std::string& vertex_shader_path);

  // Loads a fragment shader from a file. Returns true if
  // successful, and false otherwise.
  // Parameters:
  //   fragment_shader_path  The filepath for the fragment shader.
  bool LoadFragmentShaderFromFile(const std::string& fragment_shader_path);

  // This function executes the following steps:
  // 1. Compiles the vertex shader. If an error occurrs, the error information
  //    log is copied into error_info_log pointer.
  // 2. Compiles the fragment shader. If an error occurrs, the error information
  //    log is copied into error_info_log pointer.
  // 3. Links the shaders to form a shader program. If an error occurrs, the
  //    error information log is copied into error_info_log pointer.
  // 4. Cleans up temporary variables.
  // The function returns false when the creation of the program fails, and
  // returns true otherwise.
  //
  // Parameters:
  //  error_info_log  A pointer to a string that holds the error log.
  bool Create(std::string* error_info_log);

  // This function activates the shader as the current one in OpenGL.
  // Returns true if the function successfully activates the shader program.
  bool Use() const {
    if (created_) {
      // We set the shader program as active.
      glUseProgram(shader_program_id_);
      return true;
    }
    return false;
  }

 protected:
  // Compiles the vertex shader.
  bool BuildVertexShader(std::string* info_log);
  // Compiles the fragment shader.
  bool BuildFragmentShader(std::string* info_log);
  // Links the shaders to form a shader program.
  bool LinkProgram(std::string* info_log);

 private:
  // Vertex shader program source.
  std::string vertex_shader_src_;
  // Fragment shader program source.
  std::string fragment_shader_src_;
  // Vertex shader id.
  GLuint vertex_shader_;
  // Fragment shader id.
  GLuint fragment_shader_;
  // Program shader id.
  GLuint shader_program_id_;
  // Created state variable. True when this shader program is created, and false
  // otherwise.
  bool created_;
};

}  // namespace wvu

#endif  // GLUTILS_SHADER_PROGRAM_H_
