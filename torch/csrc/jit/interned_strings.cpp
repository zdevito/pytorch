#include <vector>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <mutex>
#include <sstream>
#include "ATen/optional.h"
#include "torch/csrc/assertions.h"
#include "torch/csrc/jit/interned_strings.h"
#include "string.h"
#include <iostream>

namespace torch { namespace jit {

struct InternedStrings {
  InternedStrings()
  :sym_to_ns_(static_cast<size_t>(_keys::num_symbols)) {
    #define REGISTER_SYMBOL(s) \
      string_to_sym_[#s] = namespaces::s; \
      sym_to_string_[namespaces::s] = #s; \
      sym_to_ns_[namespaces::s] = namespaces::none;

    FORALL_BASE_SYMBOLS(REGISTER_SYMBOL)
    #undef REGISTER_SYMBOL
    #define REGISTER_SYMBOL(n, s) \
      string_to_sym_[#n "::" #s] = n::s; \
      sym_to_string_[n::s] = #n "::" #s; \
      sym_to_ns_[n::s] = namespaces::n;

    FORALL_NS_SYMBOLS(REGISTER_SYMBOL)
    #undef REGISTER_SYMBOL
  }
  Symbol symbol(const std::string & s) {
    std::lock_guard<std::mutex> guard(mutex_);
    return _symbol(s);
  }
  std::pair<const char *, size_t> string(Symbol sym) {
    // Builtin Symbols are also in the maps, but
    // we can bypass the need to acquire a lock
    // to read the map for Builtins because we already
    // know their string value
    switch(sym) {
      #define DEFINE_CASE(s) \
        case namespaces::s: return {#s, strlen(#s)};
      FORALL_BASE_SYMBOLS(DEFINE_CASE)
      #undef DEFINE_CASE
      #define DEFINE_CASE(ns, s) \
        case ns::s: return {#ns "::" #s, strlen(#ns "::" #s)};
      FORALL_NS_SYMBOLS(DEFINE_CASE)
      #undef DEFINE_CASE
        default:
          return customString(sym);
    }
  }
  Symbol ns(Symbol sym) {
    switch(sym) {
      #define DEFINE_CASE(s) \
        case namespaces::s: return namespaces::none;
      FORALL_BASE_SYMBOLS(DEFINE_CASE)
      #undef DEFINE_CASE
      #define DEFINE_CASE(ns, s) \
        case ns::s: return namespaces::ns;
      FORALL_NS_SYMBOLS(DEFINE_CASE)
      #undef DEFINE_CASE
        default: {
          std::lock_guard<std::mutex> guard(mutex_);
          return sym_to_ns_.at(sym);
        }
    }
  }
private:
  // prereq - holding mutex_
  Symbol _symbol(const std::string & s) {
    auto it = string_to_sym_.find(s);
    if(it != string_to_sym_.end())
      return it->second;
    Symbol sym(sym_to_ns_.size());
    string_to_sym_[s] = sym;
    sym_to_string_[sym] = s;
    sym_to_ns_.push_back(namespaceFromString(s));
    return sym;
  }
  // prereq - holding mutex_
  Symbol namespaceFromString(const std::string& s) {
    auto pos = s.find("::");
    if(pos == std::string::npos) {
      return namespaces::none;
    }
    return _symbol(s.substr(0, pos));
  }
  std::pair<const char *, size_t> customString(Symbol sym) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = sym_to_string_.find(sym);
    JIT_ASSERT(it != sym_to_string_.end());
    return {it->second.c_str(), it->second.size()};
  }
  std::unordered_map<std::string, Symbol> string_to_sym_;
  std::unordered_map<Symbol, std::string> sym_to_string_;
  std::vector<Symbol> sym_to_ns_;

  std::mutex mutex_;
};

static InternedStrings & globalStrings() {
  static InternedStrings s;
  return s;
}

Symbol Symbol::fromQualString(const std::string & s) {
  return globalStrings().symbol(s);
}

const char * Symbol::toUnqualString() const {
  Symbol n = ns();
  if(n == namespaces::none) {
    return toQualString();
  }
  return toQualString() + globalStrings().string(n).second + 2 /* double colon */;
}

const char * Symbol::toQualString() const {
  return globalStrings().string(*this).first;
}

const char * Symbol::toDisplayString() const {
  // TODO: Make this actually return something that's "user friendly".
  // The trouble is that, for this to be usable in printf-style assert
  // statements, this has to return a const char* (whose lifetime is
  // global), so we can't actually assemble a string on the fly.
  return toQualString();
}

Symbol Symbol::ns() const {
  return globalStrings().ns(*this);
}

std::string Symbol::domainString() const {
  return domain_prefix + ns().toQualString();
}

Symbol Symbol::fromDomainAndUnqualString(const std::string & d, const std::string & s) {
  std::string qualString = d.substr(domain_prefix.size()) + "::" + s;
  return fromQualString(qualString);
}

}}
