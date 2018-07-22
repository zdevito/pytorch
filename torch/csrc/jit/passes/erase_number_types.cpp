#include "torch/csrc/jit/passes/erase_number_types.h"
#include "torch/csrc/jit/constants.h"

namespace torch { namespace jit {

static bool isNumberTypeCast(const Value* value, const Use& use) {
  auto* node = use.user;
  if (node->kind() != aten::type_as) {
    return false;
  }
  return node->inputs()[0] == value;
}

static void EraseNumberTypesOnBlock(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto sub : it->blocks()) {
      EraseNumberTypesOnBlock(sub);
    }
    switch (it->kind()) {
      case prim::Constant: {
        // remove primitive constants, replacing with tensor equivalent
        if(it->output()->type()->isSubtypeOf(*NumberType::get())) {
          auto s = *constant_as<at::Scalar>(it->output());
          WithInsertPoint guard(*it);
          Value* r = createConstant(*block->owningGraph(), s.toTensor());
          it->output()->replaceAllUsesWith(r);
        }
      } break;
      case prim::TensorToNum: {
        it->output()->replaceAllUsesWith(it->inputs()[0]);
        // Let DCE cleanup
      } break;
      case prim::NumToTensor: {
        auto* ten = it->output();
        for (const auto& use : ten->uses()) {
          if (isNumberTypeCast(ten, use)) {
            use.user->output()->replaceAllUsesWith(ten);
          }
        }
        ten->replaceAllUsesWith(it->inputs()[0]);
        // Let DCE cleanup
      } break;
      default: {
        for(auto o : it->outputs()) {
          if (o->type()->isSubtypeOf(*NumberType::get())) {
            o->setType(DynamicType::get());
          }
        }
      } break;
    }
  }
}

void EraseNumberTypes(const std::shared_ptr<Graph>& graph) {
  EraseNumberTypesOnBlock(graph->block());
}

}}
