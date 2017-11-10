#pragma once
#include <memory>
#include <vector>

namespace at {
  struct Tensor;
}
namespace torch { namespace jit {

// The interpreter run Graphs with Tensor inputs and Tensor outputs
// a separate component in the autograd handles unwrapping and wrapping
// variable objects for use in the interpreter.

struct FunctionImpl;
struct InterpreterImpl;
struct Graph;

struct Function {
  Function(std::shared_ptr<Graph> & graph);
  ~Function();
private:
  std::shared_ptr<FunctionImpl> pImpl;
  friend class InterpreterImpl;
};

struct Interpreter {
  Interpreter(const Function & function);
  void runOneStage(
    const std::vector<at::Tensor> & inputs,
    std::vector<at::Tensor> & outputs);
  ~Interpreter();
private:
  std::shared_ptr<InterpreterImpl> pImpl;
};

}}
