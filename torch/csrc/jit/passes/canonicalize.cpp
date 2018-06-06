#include "torch/csrc/jit/passes/canonicalize.h"

namespace torch { namespace jit {

// Canonicalize a graph, renumbering it so that all structurally equivalent
// graphs have same numbers.
std::shared_ptr<Graph> Canonicalize(const std::shared_ptr<Graph>& graph) {
  // copying the graph is sufficient to achieve this property in our
  // current implementation since it assigns sequential unique numbers to
  // all values copied
  return graph->copy();
}

}}
