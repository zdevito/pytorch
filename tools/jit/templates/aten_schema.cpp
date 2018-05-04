#include "torch/csrc/jit/aten_schema.h"
#include "torch/csrc/jit/tensor_conversions.h"

namespace torch { namespace jit {

using SchemaMap = std::unordered_map<std::string, std::vector<OperatorSchema>>;


std::vector<OperatorSchema> createOperatorSchemas() {
  using namespace at; // for tensor initialization
  std::vector<OperatorSchema> schemas;
  const char* names[] = {
    ${names}
  };
  at::optional<at::Tensor> tensors[] = {
    ${tensors}
  };
  at::optional<AttributeKind> attributes[] = {
    ${attributes}
  };

  using ArgumentSpec = uint32_t[4];
  ArgumentSpec arguments[] = {
    ${arguments}
  };
  using OperatorSpec = uint32_t[3];

  OperatorSpec operators[] = {
    ${operators}
  };
  size_t n_operators = ${n_operators};

  size_t next_argument = 0;

  auto getArgumentList = [&](uint32_t N){
    std::vector<Argument> result;
    for(size_t i = 0; i < N; ++i) {
      auto & a = arguments[next_argument++];
      result.push_back({ names[a[0]], tensors[a[1]], attributes[a[2]], a[3] != 0 });
    }
    return result;
  };

  for(size_t i = 0; i < n_operators; ++i) {
    auto & op = operators[i];
    schemas.push_back({names[op[0]], getArgumentList(op[1]), getArgumentList(op[2])});
  }
  return schemas;
}

std::vector<OperatorSchema> & getOperatorSchemas() {
  static std::vector<OperatorSchema> schema = createOperatorSchemas();
  return schema;
}

static SchemaMap createSchemaMap() {
  auto schemas = getOperatorSchemas();
  SchemaMap result;
  for(auto & schema : schemas) {
    auto it = result.find(schema.name);
    if(it == result.end()) {
      it = result.insert({schema.name, {}}).first;
    }
    it->second.push_back(std::move(schema));
  }
  return result;
}

const std::vector<OperatorSchema>& getOperatorSchema(const std::string& name) {
  static SchemaMap map = createSchemaMap();
  static std::vector<OperatorSchema> empty;
  auto it = map.find(name);
  if(it != map.end())
    return it->second;
  return empty;
}



}}
