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

#include <algorithm>
#include <functional>
#include <vector>
#include <string>

#include "TC_CORE.hpp"

using namespace std;
using namespace VPF;

namespace VPF {
struct TaskImpl {
  string name;
  vector<Token *> inputs;
  vector<Token *> outputs;

  p_sync_call call;
  void *args;

  TaskImpl() = delete;
  TaskImpl(const TaskImpl &other) = delete;
  TaskImpl &operator=(const TaskImpl &other) = delete;

  TaskImpl(const char *str_name, uint32_t num_inputs, uint32_t num_outputs,
           p_sync_call sync_call, void *p_args)
      : name(str_name), inputs(num_inputs), outputs(num_outputs),
        call(sync_call), args(p_args) {}
};
} // namespace VPF

Task::Task(const char *str_name, uint32_t num_inputs, uint32_t num_outputs,
           p_sync_call sync_call, void *p_args)
    : p_impl(new TaskImpl(str_name, num_inputs, num_outputs, sync_call, p_args)) {}

TaskExecStatus Task::Run() { return TaskExecStatus::TASK_EXEC_SUCCESS; }

TaskExecStatus Task::Execute() {
  auto const ret = Run();
  if (p_impl->call && p_impl->args) {
    p_impl->call(p_impl->args);
  }

  return ret;
}

bool Task::SetInput(Token *p_input, uint32_t num_input) {
  if (num_input < p_impl->inputs.size()) {
    p_impl->inputs[num_input] = p_input;
    return true;
  }

  return false;
}

void Task::ClearInputs() {
  for (auto i = 0U; i < GetNumInputs(); i++) {
    SetInput(nullptr, i);
  }
}

Token *Task::GetInput(uint32_t num_input) {
  if (num_input < p_impl->inputs.size()) {
    return p_impl->inputs[num_input];
  }

  return nullptr;
}

bool Task::SetOutput(Token *p_output, uint32_t num_output) {
  if (num_output < p_impl->outputs.size()) {
    p_impl->outputs[num_output] = p_output;
    return true;
  }
  return false;
}

void Task::ClearOutputs() {
  for (auto i = 0U; i < GetNumOutputs(); i++) {
    SetOutput(nullptr, i);
  }
}

Token *Task::GetOutput(uint32_t num_output) {
  if (num_output < p_impl->outputs.size()) {
    return p_impl->outputs[num_output];
  }

  return nullptr;
}

const char* Task::GetName() const { return p_impl->name.c_str(); }

Task::~Task() { delete p_impl; }

size_t Task::GetNumOutputs() const { return p_impl->outputs.size(); }

size_t Task::GetNumInputs() const { return p_impl->inputs.size(); }
