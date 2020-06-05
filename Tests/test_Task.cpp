/*
 * Copyright 2017-2019 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include "TC_CORE.hpp"
#include "gtest/gtest.h"

using namespace VPF;
using namespace std;

class test_Task : public ::testing::Test {
public:
  test_Task() {}
  ~test_Task() {}
  void SetUp() {}
  void TearDown() {}
};

class testTask : public Task {
public:
  testTask()
      : Task("testTask", testTask::GetNumInputs(), testTask::GetNumOutputs()) {}
  static uint32_t GetInputsNum() { return 2U; }
  static uint32_t GetOutputsNum() { return 2U; }
  TaskExecStatus Execute() { return TaskExecStatus::TASK_EXEC_SUCCESS; }
};

class test_Token : public Token {
public:
  test_Token() = default;
  ~test_Token() = default;

  size_t Retain() { return 1U; }
  size_t Release() { return 1U; }
  void *GetRawMemPtr() { return nullptr; }
};

TEST_F(test_Task, SetInput) {
  auto pA = make_shared<testTask>();
  test_Token token;

  /* Valid input, valid number;
   */
  auto res = pA->SetInput(&token, 0);
  ASSERT_EQ(true, res);

  /* Valid input, invalid number;
   */
  res = pA->SetInput(&token, pA->GetInputsNum());
  ASSERT_EQ(false, res);

  /* Valid input, not unique;
   */
  res = pA->SetInput(&token, pA->GetInputsNum() - 1);
  ASSERT_EQ(false, res);

  /* Null input, not unique;
   * Should return true as Task may have multiple inputs, each not set;
   */
  res = pA->SetInput(nullptr, 0);
  res = pA->SetInput(nullptr, pA->GetInputsNum() - 1);
  ASSERT_EQ(true, res);
}

TEST_F(test_Task, SetOutput) {
  auto pA = make_shared<testTask>();
  test_Token token;

  /* Valid output, valid number;
   */
  auto res = pA->SetOutput(&token, 0);
  ASSERT_EQ(true, res);

  /* Valid output, invalid number;
   */
  res = pA->SetOutput(&token, pA->GetOutputsNum() + 1);
  ASSERT_EQ(false, res);

  /* Valid output, not unique;
   */
  res = pA->SetOutput(&token, pA->GetOutputsNum() - 1);
  ASSERT_EQ(false, res);

  /* Null output, valid number;
   * Should return true as this method may be used to clear output before
   * generating it over;
   */
  res = pA->SetOutput(nullptr, 0);
  ASSERT_EQ(true, res);
}

TEST_F(test_Task, GetInput) {
  auto pA = make_shared<testTask>();

  /* No input set before calling method;
   */
  auto input = pA->GetInput(pA->GetInputsNum() - 1);
  EXPECT_EQ(nullptr, input);

  test_Token token;
  pA->SetInput(&token, pA->GetInputsNum() - 1);

  /* Valid input number;
   */
  input = pA->GetInput(pA->GetInputsNum() - 1);
  EXPECT_EQ(input, &token);

  /* Invalid input number;
   */
  input = pA->GetInput(pA->GetInputsNum());
  EXPECT_EQ(input, nullptr);
}

TEST_F(test_Task, GetOutput) {
  auto pA = make_shared<testTask>();

  /* No output set before calling method;
   */
  auto output = pA->GetOutput(pA->GetInputsNum() - 1);
  EXPECT_EQ(nullptr, output);

  test_Token token;
  pA->SetOutput(&token, pA->GetInputsNum() - 1);

  /* Valid output number;
   */
  output = pA->GetOutput(pA->GetInputsNum() - 1);
  EXPECT_EQ(output, &token);

  /* Invalid output number;
   */
  output = pA->GetOutput(pA->GetInputsNum());
  EXPECT_EQ(output, nullptr);
}

TEST_F(test_Task, GetNumOutputs) {
  auto pA = make_shared<testTask>();
  test_Token token;

  auto numOutputs = pA->GetNumOutputs();
  EXPECT_EQ(pA->GetOutputsNum(), numOutputs);

  /* Setting particular output shouldn't affect the result;
   */
  pA->SetOutput(&token, 0U);
  numOutputs = pA->GetNumOutputs();
  EXPECT_EQ(pA->GetOutputsNum(), numOutputs);
}

TEST_F(test_Task, GetNumInputs) {
  auto pA = make_shared<testTask>();
  test_Token token;

  auto numInputs = pA->GetNumInputs();
  EXPECT_EQ(pA->GetInputsNum(), numInputs);

  /* Setting particular input shouldn't affect the result;
   */
  pA->SetInput(&token, 0U);
  numInputs = pA->GetNumInputs();
  EXPECT_EQ(pA->GetInputsNum(), numInputs);
}
