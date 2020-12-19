#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/python.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <torchpy.h>

void compare_torchpy_jit(const char* model_filename) {
  // Test
  std::ostringstream pkg;
  pkg << "torchpy/example/generated/" << model_filename;
  std::ostringstream jit;
  jit << "torchpy/example/generated/" << model_filename << "_jit";
  torch::InterpreterManager m(1);
  torch::Package p = m.load_package(pkg.str());
  auto model = p.load_pickle("model", "model.pkl");
  at::IValue eg;
  {
    auto I = p.acquire_session();
    eg = I.self.attr("load_pickle")({"model", "example.pkl"}).toIValue();
  }

  at::Tensor output = model(eg.toTuple()->elements()).toTensor();

  // Reference
  auto ref_model = torch::jit::load(jit.str());
  at::Tensor ref_output =
      ref_model.forward(eg.toTuple()->elements()).toTensor();

  ASSERT_TRUE(ref_output.allclose(output, 1e-03, 1e-05));
}

TEST(TorchpyTest, SimpleModel) {
  compare_torchpy_jit("simple");
}

TEST(TorchpyTest, ResNet) {
  compare_torchpy_jit("resnet");
}

TEST(TorchpyTest, Movable) {
  torch::InterpreterManager m(1);
  torch::MovableObject obj;
  {
    auto I = m.acquire_one();
    auto model =
        I.global("torch.nn", "Module")(std::vector<torch::PythonObject>());
    obj = I.create_movable(model);
  }
  obj.acquire_session();
}