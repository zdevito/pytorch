#pragma once
#include "ATen/ATen.h"

#include "torch/csrc/jit/type.h"
#include "torch/csrc/jit/ivalue.h"

namespace torch { namespace jit {

// schema as used in the compiler for resolving function calls and reporting
// errors. These objects should be constructed from C10 schema once those
// are available.

struct Argument {
  Argument(
      std::string name = "",
      TypePtr type = nullptr,
      c10::optional<int32_t> N = c10::nullopt,
      c10::optional<IValue> default_value = c10::nullopt,
      bool kwarg_only = false)
      : name_(std::move(name)),
        annotated_type_(type ? type : DynamicType::get()),
        type_(getUnannotatedType(annotated_type_)),
        N_(std::move(N)),
        default_value_(std::move(default_value)),
        kwarg_only_(kwarg_only) {}
  const std::string& name() const {
    return name_;
  }
  TypePtr type() const {
    return type_;
  }
  // type with possible additional aliasing information added
  TypePtr annotatedType() const {
    return annotated_type_;
  }
  c10::optional<int32_t> N() const {
    return N_;
  }
  c10::optional<IValue> default_value() const {
    return default_value_;
  }
  bool kwarg_only() const {
    return kwarg_only_;
  }
private:
  static TypePtr getUnannotatedType(TypePtr typ) {
    if(typ->kind() == TypeKind::AnnotatedType) {
      auto elem = typ->expect<AnnotatedType>()->getElementType();
      return getUnannotatedType(elem);
    }
    return typ->withContained(fmap(typ->containedTypes(), getUnannotatedType));
  }
  std::string name_;
  TypePtr annotated_type_;
  TypePtr type_;
  // for list types, an optional statically known length for the list
  // e.g. for int[3]: type = ListType::ofInts(), N = 3
  // If present, this will allow scalars to be broadcast to this length to
  // become a list.
  c10::optional<int32_t> N_;

  c10::optional<IValue> default_value_;
  // is this only specifyable as a keyword argument?
  bool kwarg_only_;
};

struct FunctionSchema {
  FunctionSchema(
      std::string name,
      std::vector<Argument> arguments,
      std::vector<Argument> returns,
      bool is_vararg = false,
      bool is_varret = false,
      std::vector<std::string> writes = {})
      : name_(std::move(name)),
        arguments_(std::move(arguments)),
        returns_(std::move(returns)),
        is_vararg_(is_vararg),
        is_varret_(is_varret),
        writes_(std::move(writes)) {
    validate();
  }
  FunctionSchema(
      Symbol name,
      std::vector<Argument> arguments,
      std::vector<Argument> returns,
      bool is_vararg = false,
      bool is_varret = false,
      std::vector<std::string> writes = {})
      : FunctionSchema(
            name.toQualString(),
            std::move(std::move(arguments)),
            std::move(std::move(returns)),
            is_vararg,
            is_varret) {
    validate();
  }
private:
  const std::string name_;
  const std::vector<Argument> arguments_;
  const std::vector<Argument> returns_;
  // if true then this schema takes an arbitrary number of additional arguments
  // after the argument specified in arguments
  // currently this is used primarily to represent 'primtive' operators whose
  // arguments are not checked by schema
  const bool is_vararg_;
  const bool is_varret_;

  // set of alias sets in Arguments that are written to by this op
  const std::vector<std::string> writes_;
public:
  const std::string& name() const {
    return name_;
  }
  const std::vector<Argument>& arguments() const {
    return arguments_;
  }
  const std::vector<Argument>& returns() const {
    return returns_;
  }
  const std::vector<std::string>& writes() const {
    return writes_;
  }
  bool is_vararg() const {
    return is_vararg_;
  }
  bool is_varret() const {
    return is_varret_;
  }
  bool has_world_token() const {
    // todo: mutable if writes_.size() > 0;
    return arguments().size() > 0 && arguments().front().type() == WorldType::get();
  }
  bool is_mutable() const {
    return writes().size() > 0;
  }
  c10::optional<int> argumentIndexWithName(const std::string& name) const {
    for(size_t i = 0; i < arguments().size(); ++i) {
      if(name == arguments()[i].name())
        return i;
    }
    return c10::nullopt;
  }

 private:
  void validate() const {
    if(has_world_token()) {
      JIT_ASSERT(returns().at(0).type() == WorldType::get());
    }
  }
};

// for debugging, make sure we can describe the call site
inline std::ostream& operator<<(std::ostream& out, const Argument& arg) {
  return out << arg.type()->str() << " " << arg.name() << (arg.default_value() ? "=<default>" : "");
}

inline std::ostream& operator<<(std::ostream& out, const FunctionSchema& schema) {
  // eventually this should look almost identical to python arg parser, but
  // it is simpler for now to work directly on this schema

  out << schema.name();
  out << "(";

  bool seen_kwarg_only = false;
  for(size_t i = 0; i < schema.arguments().size(); ++i) {
    if (i > 0) out << ", ";
    if (schema.arguments()[i].kwarg_only() && !seen_kwarg_only) {
      out << "*, ";
      seen_kwarg_only = true;
    }
    out << schema.arguments()[i];
  }

  out << ") -> ";
  if (schema.returns().size() == 1) {
    out << schema.returns().at(0).type()->str();
  } else if (schema.returns().size() > 1) {
    out << "(";
    for (size_t i = 0; i < schema.returns().size(); ++i) {
      if (i > 0) out << ", ";
      out << schema.returns()[i].type()->str();
    }
    out << ")";
  }
  return out;
}

}}
