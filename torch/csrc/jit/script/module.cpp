#include <torch/csrc/jit/script/module.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/export.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/script/compiler.h>
#include <torch/csrc/jit/script/error_report.h>
#include <torch/csrc/jit/script/schema_matching.h>

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

} // namespace script
} // namespace jit
} // namespace torch
