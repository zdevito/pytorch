#include "torch/csrc/jit/constants.h"
#include "torch/csrc/jit/operator.h"

namespace torch { namespace jit {

Value* createConstant(
    Graph& g,
    IValue val,
    at::optional<script::SourceRange> loc,
    TypePtr typ) {
  at::Tensor ref = std::move(val).toTensor();
  JIT_ASSERT(ref.defined());
  auto n = g.create(prim::Constant);
  n->t_(attr::value, ref.clone());
  n->output()->inferTypeFrom(ref);
  if(loc)
    n->setSourceLocation(std::make_shared<script::SourceRange>(*loc));
  if(typ)
    n->output()->setType(typ);
  return g.insertNode(n)->output();
}

RegisterOperators reg({
  Operator(
      prim::Constant,
      [](Node* node) {
        auto t = autograd::make_variable(node->t(attr::value));
        return [t](Stack& stack) {
          stack.push_back(t);
          return 0;
        };
      }),
});

}}
