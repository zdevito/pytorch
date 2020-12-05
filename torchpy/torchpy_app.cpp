#define _GNU_SOURCE
#include <dlfcn.h>

#include <pthread.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

#include <assert.h>
#include "torchpy.h"

#include <ATen/ATen.h>
#include <ATen/TypeDefault.h>

#include <torch/script.h>

#include <Python.h>

typedef void (*function_type)(const char*);

constexpr auto latency_p = {
    25.,
    50.,
    95.}; //{1., 5., 25., 50., 75., 90., 95., 99., 99.25, 99.5, 99.75, 99.9};

struct Report {
  std::string benchmark;
  bool jit;
  size_t n_threads;
  size_t n_interp;
  size_t items_completed;
  double work_items_per_second;
  std::vector<double> latencies;
  static void report_header(std::ostream& out) {
    out << "benchmark, jit, n_threads, interpreter_strategy, work_items_completed, work_items_per_second";
    for (double l : latency_p) {
      out << ", p" << l << "_latency";
    }
    out << "\n";
  }
  void report(std::ostream& out) {
    const char* it = nullptr;
    if (n_interp == 1) {
      it = "one_global_interpreter";
    } else if (n_interp == n_threads) {
      it = "one_interpreter_per_thread";
    } else {
      it = "one_intepreter_per_two_threads";
    }

    out << benchmark << ", " << jit << ", " << n_threads << ", " << it << ", "
        << items_completed << ", " << work_items_per_second;
    for (double l : latencies) {
      out << ", " << l;
    }
    out << "\n";
  }
};

const int min_items_to_complete = 1;

struct Benchmark {
  Benchmark(
      torch::InterpreterManager& manager,
      size_t n_threads,
      size_t n_interpreters,
      std::string file_to_run,
      bool jit,
      size_t n_seconds = 1)
      : manager_(manager),
        n_threads_(n_threads),
        n_interpreters_(n_interpreters),
        file_to_run_(file_to_run),
        jit_(jit),
        n_seconds_(n_seconds),
        should_run_(true),
        items_completed_(0),
        reached_min_items_completed_(0) {
    manager.debugLimitInterpreters(n_interpreters_);
  }

  Report run() {
    pthread_barrier_init(&first_run_, nullptr, n_threads_ + 1);

    torch::Package package = manager_.load_package(file_to_run_);

    std::vector<at::IValue> eg;
    {
      auto I = package.acquire_session();

      eg = I.global("builtins", "tuple")(
                I.self.attr("load_pickle")({"model", "example.pkl"}))
               .toIValue()
               .toTuple()
               ->elements();
    }
    for (at::IValue& iv : eg) {
      if (iv.isTensor()) {
        at::Tensor t = iv.toTensor();
        t.unsafeGetTensorImpl()->pyobj();
        ;
      }
    }
    torch::jit::Module model_jit;
    if (jit_) {
      model_jit = torch::jit::load(file_to_run_ + "_jit");
      run_one_work_item = [&model_jit, &eg](int i) { model_jit.forward(eg); };
    } else {
      struct Run {
        Run(torch::MovableObject a,
            std::vector<at::IValue>& b,
            const torch::Interpreter* i)
            : obj(std::move(a)), eg(b), interps(i) {}
        void operator()(int i) {
          auto I = obj.acquire_session(&interps[i]);
          I.self(eg);
        }
        torch::MovableObject obj;
        std::vector<at::IValue>& eg;
        const torch::Interpreter* interps;
      };
      auto mo = package.load([&](torch::InterpreterSession& I) {
        auto obj = I.self.attr("load_pickle")({"model", "model.pkl"});
        return I.global("gpu_wrapper", "GPUWrapper")({obj});
      });
      run_one_work_item =
          Run(std::move(mo), eg, manager_.all_instances().data());
    }

    std::vector<std::vector<double>> latencies(n_threads_);

    for (size_t i = 0; i < n_threads_; ++i) {
      threads_.emplace_back([this, &latencies, i] {
        // do initial work
        run_one_work_item(i);

        pthread_barrier_wait(&first_run_);
        size_t local_items_completed = 0;
        while (should_run_) {
          auto begin = std::chrono::steady_clock::now();
          run_one_work_item(i);
          auto end = std::chrono::steady_clock::now();
          double work_seconds =
              std::chrono::duration<double>(end - begin).count();
          latencies[i].push_back(work_seconds);
          local_items_completed++;
          if (local_items_completed == min_items_to_complete) {
            reached_min_items_completed_++;
          }
        }
        items_completed_ += local_items_completed;
      });
    }

    pthread_barrier_wait(&first_run_);
    auto begin = std::chrono::steady_clock::now();
    auto try_stop_at = begin + std::chrono::seconds(n_seconds_);
    std::this_thread::sleep_until(try_stop_at);
    for (int i = 0; reached_min_items_completed_ < n_interpreters_; ++i) {
      std::this_thread::sleep_until(begin + (i+ 2)*std::chrono::seconds(n_seconds_));
    }
    should_run_ = false;
    for (std::thread& thread : threads_) {
      thread.join();
    }
    auto end = std::chrono::steady_clock::now();
    double total_seconds = std::chrono::duration<double>(end - begin).count();
    Report report;
    report.benchmark = file_to_run_;
    report.jit = jit_;
    report.n_interp = n_interpreters_;
    report.n_threads = n_threads_;
    report.items_completed = items_completed_;
    report.work_items_per_second = items_completed_ / total_seconds;
    reportLatencies(report.latencies, latencies);
    run_one_work_item = nullptr;
    return report;
  }

 private:
  void reportLatencies(
      std::vector<double>& results,
      const std::vector<std::vector<double>>& latencies) {
    std::vector<double> flat_latencies;
    for (const auto& elem : latencies) {
      flat_latencies.insert(flat_latencies.end(), elem.begin(), elem.end());
    }
    std::sort(flat_latencies.begin(), flat_latencies.end());
    for (double target : latency_p) {
      size_t idx = size_t(flat_latencies.size() * target / 100.0);
      double time = flat_latencies.size() == 0
          ? 0
          : flat_latencies.at(std::min(flat_latencies.size() - 1, idx));
      results.push_back(time);
    }
  }
  torch::InterpreterManager& manager_;
  size_t n_threads_;
  size_t n_interpreters_;
  std::string file_to_run_;
  bool jit_;
  size_t n_seconds_;
  pthread_barrier_t first_run_;
  std::atomic<bool> should_run_;
  std::atomic<size_t> items_completed_;
  std::atomic<size_t> reached_min_items_completed_;
  std::vector<std::thread> threads_;
  std::function<void(int)> run_one_work_item;
};

int main(int argc, char* argv[]) {
  // make sure things work even when python exists in the main app
  Py_Initialize();
  torch::InterpreterManager manager(40);
  auto n_threads = {1, 2, 4, 8, 16, 32, 40};
  Report::report_header(std::cout);
  for (int i = 1; i < argc; ++i) {
    std::string model_file = argv[i];
    for (int n_thread : n_threads) {
      size_t prev = 0;
      auto interpreter_strategy = {
          n_thread}; // {1, std::max<int>(1, n_thread / 2), n_thread};
      for (int n_interp : interpreter_strategy) {
        for (bool jit : {false}) {
          if (jit) {
            std::fstream jit_file(model_file + "_jit");
            if (!jit_file.good()) {
              continue; // no jit file present
            }
          }
          Benchmark b(manager, n_thread, n_interp, model_file, jit);
          Report r = b.run();
          prev = n_interp;
          r.report(std::cout);
        }
      }
    }
  }

  return 0;
}