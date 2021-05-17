/*
 * Copyright 2019 NVIDIA Corporation
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *    http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "Version.hpp"
#include <cstdint>
#include <utility>

#if defined(_WIN32)
#define DllExport __declspec(dllexport)
#else
#define DllExport
#endif

namespace VPF {

/* Interface for data exchange;
 * It represents memory object (CPU- or GPU-side memory etc.);
 */
class DllExport Token {
public:
  Token &operator=(const Token &other) = delete;
  Token(const Token &other) = delete;

  virtual ~Token();

protected:
  Token();
};

enum class TaskExecStatus { TASK_EXEC_SUCCESS, TASK_EXEC_FAIL };

/* Synchronization call which will be done after a blocking task;
 */
typedef void (*p_sync_call)(void *p_args);

/* Task is unit of processing; Inherit from this class to add user-defined
 * processing stage;
 */
class DllExport Task {
public:
  Task() = delete;
  Task(const Task &other) = delete;
  Task &operator=(const Task &other) = delete;

  virtual ~Task();

  /* Method to be overridden in ancestors;
   */
  virtual TaskExecStatus Run();

  /* Call this method to run the task;
   */
  virtual TaskExecStatus Execute();

  /* Sets given token as input;
   * Doesn't take ownership of object passed by pointer, only stores it
   * within inplementation;
   */
  bool SetInput(Token *input, uint32_t input_num);

  /* Sets given token as output;
   * Doesn't take ownership of object passed by pointer, only stores it
   * within inplementation;
   */
  bool SetOutput(Token *output, uint32_t output_num);

  /* Sets all inputs to nullptr;
   */
  void ClearInputs();

  /* Sets all outputs to nullptr;
   */
  void ClearOutputs();

  /* Returns pointer to task input in case of success, nullptr otherwise;
   */
  Token *GetInput(uint32_t num_input = 0);

  /* Returns pointer to task output in case of success, nullptr otherwise;
   */
  Token *GetOutput(uint32_t num_output = 0);

  /* Returns number of outputs;
   */
  uint64_t GetNumOutputs() const;

  /* Returns number of inputs;
   */
  uint64_t GetNumInputs() const;

protected:
  Task(const char *str_name, uint32_t num_inputs, uint32_t num_outputs,
       p_sync_call sync_call = nullptr, void *p_args = nullptr);

  /* Hidden implementation;
   */
  struct TaskImpl *p_impl = nullptr;
};
} // namespace VPF