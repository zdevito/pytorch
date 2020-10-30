#include <gtest/gtest.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <torchpy.h>
#include <unistd.h>
#include <future>

TEST(TorchpyTest, MultiSerialSimpleModel) {
  torch::InterpreterManager manager(3);
  torch::Package p = manager.load_package("torchpy/example/generated/simple");
  auto model = p.load_pickle("model", "model.pkl");
  auto ref_model = torch::jit::load("torchpy/example/generated/simple_jit");

  auto input = torch::ones({10, 20});
  size_t ninterp = 3;
  std::vector<at::Tensor> outputs;

  // Futures on model forward
  for (size_t i = 0; i < ninterp; i++) {
    outputs.push_back(model({input}).toTensor());
  }

  // Generate reference
  auto ref_output = ref_model.forward({input}).toTensor();

  // Compare all to reference
  for (size_t i = 0; i < ninterp; i++) {
    ASSERT_TRUE(ref_output.equal(outputs[i]));
  }
}

TEST(TorchpyTest, ThreadedSimpleModel) {
  size_t nthreads = 3;
  torch::InterpreterManager manager(nthreads);

  torch::Package p = manager.load_package("torchpy/example/generated/simple");
  auto model = p.load_pickle("model", "model.pkl");
  auto ref_model = torch::jit::load("torchpy/example/generated/simple_jit");

  auto input = torch::ones({10, 20});

  std::vector<at::Tensor> outputs;

  // Futures on model forward
  // Futures on model forward
  std::vector<std::future<at::Tensor>> futures;
  for (size_t i = 0; i < nthreads; i++) {
    futures.push_back(std::async(std::launch::async, [&model]() {
      auto input = torch::ones({10, 20});
      for (int i = 0; i < 100; ++i) {
        model({input}).toTensor();
      }
      auto result = model({input}).toTensor();
      return result;
    }));
  }
  for (size_t i = 0; i < nthreads; i++) {
    outputs.push_back(futures[i].get());
  }

  // Generate reference
  auto ref_output = ref_model.forward({input}).toTensor();

  // Compare all to reference
  for (size_t i = 0; i < nthreads; i++) {
    ASSERT_TRUE(ref_output.equal(outputs[i]));
  }
}
