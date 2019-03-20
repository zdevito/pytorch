#if defined(USE_GTEST)
#include <gtest/gtest.h>
#endif

// To add a new test file:
// 1. Add a test_foo.h file in this directory
// 2. include test_base.h
// 3. Write your tests as pure functions starting with "test", like "testFoo"
// 4. Include test_foo.h here and add it to the appropriate macro listing
#include <test/cpp/jit/test_alias_analysis.h>
#include <test/cpp/jit/test_argument_spec.h>
#include <test/cpp/jit/test_autodiff.h>
#include <test/cpp/jit/test_class_parser.h>
#include <test/cpp/jit/test_constant_pooling.h>
#include <test/cpp/jit/test_create_autodiff_subgraphs.h>
#include <test/cpp/jit/test_fuser.h>
#include <test/cpp/jit/test_graph_executor.h>
#include <test/cpp/jit/test_ir.h>
#include <test/cpp/jit/test_irparser.h>
#include <test/cpp/jit/test_ivalue.h>
#include <test/cpp/jit/test_misc.h>
#include <test/cpp/jit/test_netdef_converter.h>
#include <test/cpp/jit/test_subgraph_utils.h>

using namespace torch::jit::script;
using namespace torch::jit::test;

namespace torch {
namespace jit {
#define TH_FORALL_TESTS(_)         \
  _(ArgSpec) \
  _(ADFormulas)                    \
  _(Attributes)                    \
  _(Blocks)                        \
  _(CodeTemplate)                  \
  _(ControlFlow)                   \
  _(CreateAutodiffSubgraphs)       \
  _(CustomOperators)               \
  _(Differentiate)                 \
  _(DifferentiateWithRequiresGrad) \
  _(DynamicDAG)                    \
  _(FromQualString)                \
  _(InternedStrings)               \
  _(IValue)                        \
  _(Proto)                         \
  _(RegisterFusionCachesKernel)    \
  _(SchemaParser)                  \
  _(TopologicalIndex)              \
  _(TopologicalMove)               \
  _(SubgraphUtils)                 \
  _(AliasAnalysis)                 \
  _(WriteTracking)                 \
  _(Wildcards)                     \
  _(MemoryDAG)                     \
  _(IRParser)                      \
  _(ConstantPooling)               \
  _(NetDefConverter)               \
  _(THNNConv)                      \
  _(ATenNativeBatchNorm)           \
  _(NoneSchemaMatch)               \
  _(ClassParser)


#define TH_FORALL_TESTS_CUDA(_) \
  _(ArgumentSpec)               \
  _(Fusion)                     \
  _(GraphExecutor)              \
  _(Interp)

#if defined(USE_GTEST)

#define JIT_GTEST(name) \
  TEST(JitTest, name) { \
    test##name();       \
  }
TH_FORALL_TESTS(JIT_GTEST)
#undef JIT_TEST

#define JIT_GTEST_CUDA(name)   \
  TEST(JitTest, name##_CUDA) { \
    test##name();              \
  }
TH_FORALL_TESTS_CUDA(JIT_GTEST_CUDA)
#undef JIT_TEST_CUDA
#endif

#define JIT_TEST(name) test##name();
void runJITCPPTests() {
  TH_FORALL_TESTS(JIT_TEST)
  TH_FORALL_TESTS_CUDA(JIT_TEST)

  // This test is special since it requires prior setup in python.
  // So it's included here but not in the pure cpp gtest suite
  testEvalModeForLoadedModule();
}
#undef JIT_TEST

} // namespace jit
} // namespace torch
