#pragma once

#include "ATen/CPUGenerator.h"

namespace at {

#define AT_ASSERT(cond, ...) if (! (cond) ) { at::runtime_error(__VA_ARGS__); }

[[noreturn]]
void runtime_error(const char *format, ...);

template <typename T, typename Base>
static inline T* checked_cast(Base* expr, const char * name, int pos) {
  if(!expr) {
    runtime_error("Expected a Tensor of type %s but found an undefined Tensor for argument #%d '%s'",
      T::typeString(),pos,name);
  }
  if(auto result = dynamic_cast<T*>(expr))
    return result;
  runtime_error("Expected object of type %s but found type %s for argument #%d '%s'",
    T::typeString(),expr->type().toString(),pos,name);
}

struct CPUGenerator;
struct Generator;
static inline CPUGenerator * check_generator(Generator* expr) {
  if(auto result = dynamic_cast<CPUGenerator*>(expr))
    return result;
  runtime_error("Expected a 'CPUGenerator' but found 'CUDAGenerator'");
}

} // at
