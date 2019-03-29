#include <torch/csrc/jit/script/module.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/export.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/script/compiler.h>
#include <torch/csrc/jit/script/error_report.h>
#include <torch/csrc/jit/script/schema_matching.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>

namespace torch {
namespace jit {
namespace script {

struct RecursiveMethodCallError : public std::exception {};
void placeholderCreator(Method&) {
  throw RecursiveMethodCallError();
}

Value* Function::try_emit_call(
    Graph& graph,
    const SourceRange& loc,
    c10::optional<NamedValue> self,
    ArrayRef<NamedValue> args,
    ArrayRef<NamedValue> kwargs,
    std::stringstream& failure_messages,
    bool conv_tensors_to_nums) {
  try {
    ensure_defined();
  } catch (RecursiveMethodCallError&) {
    throw ErrorReport(loc)
        << " method '" << name()
        << "' is called recursively involving this call site. "
        << "Recursive calls are not supported";
  }
  auto fn = this->graph();

  auto matched_schema = tryMatchSchema(
      getSchema(),
      loc,
      graph,
      std::move(self),
      args,
      kwargs,
      failure_messages,
      conv_tensors_to_nums);
  if (!matched_schema)
    return nullptr;

  check_single_output();
  return inlineCallTo(graph, *fn, matched_schema->inputs).at(0);
}

Value* Function::emit_call(
    Graph& graph,
    const SourceRange& loc,
    ArrayRef<NamedValue> args,
    ArrayRef<NamedValue> kwargs) {
  AT_ASSERT(!executor_);
  std::stringstream failure_messages;
  if (auto result = try_emit_call(
          graph,
          loc,
          c10::nullopt,
          args,
          kwargs,
          failure_messages,
          /*conv_tensors_to_nums=*/true)) {
    return result;
  }
  throw ErrorReport(loc) << failure_messages.str();
}

void Module::to(at::Device device, at::ScalarType dtype, bool non_blocking) {
  to_impl(device, dtype, non_blocking);
}

void Module::to(at::ScalarType dtype, bool non_blocking) {
  to_impl(/*device=*/c10::nullopt, dtype, non_blocking);
}

void Module::to(at::Device device, bool non_blocking) {
  to_impl(device, /*dtype=*/c10::nullopt, non_blocking);
}

void Module::save(std::ostream& out, const ExtraFilesMap& extra_files) {
  ExportModule(*this, out, extra_files);
}

void Module::save(
    const std::string& filename,
    const ExtraFilesMap& extra_files) {
  ExportModule(*this, filename, extra_files);
}

void Module::to_impl(
    const c10::optional<at::Device>& device,
    const c10::optional<at::ScalarType>& dtype,
    bool non_blocking) {
  // First call `to()` on every child module.
  for (auto& child : get_modules()) {
    child->to_impl(device, dtype, non_blocking);
  }
  // Then convert every of our parameters.
  for (auto& parameter : get_parameters()) {
    // Need to access the `at::Tensor` as a `Variable` here.
    autograd::Variable variable = parameter.value().toTensor();
    at::Tensor data = variable.data();
    // Use the data's original device or dtype if not supplied here.
    auto new_data = data.to(
        device.value_or(data.device()),
        dtype.value_or(data.scalar_type()),
        non_blocking);
    variable.set_data(new_data);
  }
}


// lower_first_class_method and lift_lowered_method are transitionary functions
// used to translate between module-as-first-class code generation,
// and module-as-special execution. Once module-as-first-class execution is
// debugged, then we can remove both and remove the lowered_functions_ table.

// remove the first module argument, replacing any access of its parameters/attributes
// with extra_ivalue input Slots that hold what value to pass into the graph
std::pair<std::shared_ptr<Graph>, std::vector<Slot>> lower_graph(
    const ModulePtr& self,
    Graph& g_) {
  std::shared_ptr<Graph> g = g_.copy();
  std::vector<Slot> extra_ivalues;
  auto self_value = g->inputs().at(0);
  std::vector<std::pair<Node*, ModulePtr>> to_scan;
  for (Use use : self_value->uses()) {
    to_scan.emplace_back(self, use.user);
  }
  std::vector<Node*> to_clean;
  while (to_scan.size() > 0) {
    Node* n;
    ModulePtr mod;
    std::tie(n, mod) = to_scan.back();
    if (n->kind() != prim::GetAttr) {
      throw ErrorReport(n->getSourceLocation())
          << "temporary: the only valid use of a module is looking up an attribute";
    }
    Slot slot(mod, mod->type()->getAttributeSlot(n->s(attr::name)));
    if (ClassTypePtr c = n->output()->type()->cast<ClassType>()) {
      if (c->name() == "Module") {
        auto obj = slot.value().toObject();
        for (Use use : n->output()->uses()) {
          to_scan.emplace_back(obj, use.user);
        }
        to_clean.emplace_back(n);
        continue;
      }
    }
    Value* v = g->addInput()->copyMetadata(n->output());
    extra_ivalues.emplace_back(std::move(slot));
    n->output()->replaceAllUsesWith(v);
    n->destroy();
    to_scan.pop_back();
  }

  while (to_clean.size() > 0) {
    Node* n = to_clean.back();
    AT_ASSERT(!n->hasUses());
    n->destroy();
    to_clean.pop_back();
  }
  AT_ASSERT(!self_value->hasUses());
  g->eraseInput(0);
  return std::make_pair(std::move(g), std::move(extra_ivalues));
}

Method* Module::lower_first_class_method(Function* fn) {
  auto lowered = lower_graph(module_object(), *fn->graph());
  create_method_from_trace(fn->name(), lowered.first, std::move(lowered.second));
}

Method* Module::lift_lowered_method(Function* fn) const {

}


} // namespace script
} // namespace jit
} // namespace torch
