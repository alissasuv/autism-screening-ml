#pragma once

#include <algorithm>
#include <cstddef>
#include <exception>
#include <thread>
#include <vector>

namespace forest_parallel {

template <typename Work>
void for_each_index(
    std::size_t tasks,
    std::size_t n_jobs,
    Work&& work
) {
    if (tasks == 0) {
        return;
    }

    const std::size_t workers = std::min(tasks, n_jobs);

    // Avoid thread creation for sequential execution.
    if (workers == 1) {
        for (std::size_t i = 0; i < tasks; ++i) {
            work(i);
        }
        return;
    }

    // Each task owns one exception slot.
    std::vector<std::exception_ptr> errors(tasks);

    {
        std::vector<std::jthread> threads;
        threads.reserve(workers);

        for (std::size_t worker = 0;
             worker < workers;
             ++worker) {
            threads.emplace_back([&, worker] {
                for (std::size_t i = worker;
                     i < tasks;
                     i += workers) {
                    try {
                        work(i);
                    } catch (...) {
                        errors[i] = std::current_exception();
                    }
                }
            });
        }
    }

    // All jthreads have joined before inspecting results.
    for (const auto& error : errors) {
        if (error) {
            std::rethrow_exception(error);
        }
    }
}

} // namespace forest_parallel


