// in memory description of all ATen Ops similar to Caffe2 schema
// once C10 exists this can be removed, or stubbed out, but we need
// it now to implement correct semantic checking for script
#pragma once
#include "ATen/ATen.h"
#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

struct Argument {
  const std::string name;
  const at::optional<at::Tensor> default_value;
  const at::optional<AttributeKind> attribute_kind;
  bool is_list;
};

struct OperatorSchema {
  const std::string name;
  const std::vector<Argument> arguments;
  const std::vector<Argument> returns;

  at::optional<Argument> argumentWithName(const std::string& name) {
    auto it = std::find_if(arguments.begin(), arguments.end(), [&](const Argument& arg) {
      return arg.name == name;
    });
    if(it == arguments.end())
      return at::nullopt;
    return *it;
  }
};

const std::vector<OperatorSchema>& getOperatorSchema(const std::string& name);

}}
