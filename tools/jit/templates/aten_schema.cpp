#include "torch/csrc/jit/aten_schema.h"


namespace torch { namespace jit {

using SchemaMap = std::unordered_map<std::string, std::vector<OperatorSchema>>;

${schemas}

static SchemaMap createSchemaMap() {
  std::vector<OperatorSchema> schemas;
  ${schemas_push_back}
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
