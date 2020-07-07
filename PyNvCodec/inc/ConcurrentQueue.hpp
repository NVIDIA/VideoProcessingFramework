/*
 * Copyright 2020 NVIDIA Corporation
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
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>

namespace VPF {
class Semaphore {
public:
  explicit Semaphore(size_t capacity) : m_counter(capacity) {}

  void Post() {
    std::unique_lock<decltype(m_mutex)> lock(m_mutex);
    ++m_counter;
    m_cond_var.notify_one();
  }

  void Wait() {
    std::unique_lock<decltype(m_mutex)> lock(m_mutex);
    while (!m_counter) {
      m_cond_var.wait(lock);
    }
    --m_counter;
  }

private:
  std::mutex m_mutex;
  std::condition_variable m_cond_var;
  size_t m_counter = 0UL;
};

template <typename T> class ConcurrentQueue {
public:
  ConcurrentQueue() = delete;
  ConcurrentQueue(const ConcurrentQueue &other) = delete;
  ConcurrentQueue &operator=(const ConcurrentQueue &other) = delete;

  explicit ConcurrentQueue(size_t capacity)
      : m_capacity(capacity), m_items(0), m_slots(capacity) {}

  virtual ~ConcurrentQueue() = default;

  void Push(const T &item) {
    m_slots.Wait();
    m_mutex.lock();
    m_queue.push(item);
    m_mutex.unlock();
    m_items.Post();
  }

  void Pop(T &item) {
    m_items.Wait();
    m_mutex.lock();
    item = m_queue.front();
    m_queue.pop();
    m_mutex.unlock();
    m_slots.Post();
  }

private:
  size_t m_capacity;
  std::mutex m_mutex;
  Semaphore m_slots;
  Semaphore m_items;
  std::queue<T> m_queue;
};

} // namespace VPF