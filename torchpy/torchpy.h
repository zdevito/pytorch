#pragma once
#include "interpreter/interpreter_impl.h"
#include <assert.h>
#include <dlfcn.h>
#include <unistd.h>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace torch {


struct InterpreterSession {
  InterpreterSession(InterpreterSessionImpl* impl, std::function<void(void)> notify_complete)
  : impl_(impl), notify_complete_(std::move(notify_complete)) {}

  PythonObject self; // when retreived from a PythonMovable this will be set.
  InterpreterSession(InterpreterSession&&) = default;
  ~InterpreterSession() {
    notify_complete_();
  }
  PythonObject global(const char* module, const char* name) {
    return impl_->global(module, name);
  }
  PythonObject from_ivalue(at::IValue ivalue) {
      return impl_->from_ivalue(std::move(ivalue));
  }

private:
  friend struct MovableObject;
  friend struct Package;
  std::unique_ptr<InterpreterSessionImpl> impl_;
  std::function<void(void)> notify_complete_;
};


inline void noop(void) {}

struct DLHandle {};

class Interpreter {
 private:
  std::string library_name_;
  void* handle_;
  std::unique_ptr<InterpreterImpl> pImpl_;

 public:
  Interpreter() : handle_(nullptr) {
    char library_name[L_tmpnam];
    library_name_ = library_name;
    std::tmpnam(library_name);
    {
      std::ifstream src("build/lib/libinterpreter.so", std::ios::binary);
      std::ofstream dst(library_name, std::ios::binary);
      dst << src.rdbuf();
    }
    handle_ = dlopen(library_name, RTLD_LOCAL | RTLD_LAZY);
    if (!handle_) {
      throw std::runtime_error(dlerror());
    }

    // technically, we can unlike the library right after dlopen, and this is
    // better for cleanup because even if we crash the library doesn't stick
    // around. However, its crap for debugging because gdb can't find the
    // symbols if the library is no longer present.
    unlink(library_name_.c_str());

    void* new_interpreter_impl = dlsym(handle_, "new_interpreter_impl");
    assert(new_interpreter_impl);
    pImpl_ = std::unique_ptr<InterpreterImpl>(((InterpreterImpl*(*)(void)) new_interpreter_impl)());
  }
  InterpreterSession acquire_session(
      std::function<void(void)> notify_complete = noop) const {
    return InterpreterSession(pImpl_->acquire_session(), std::move(notify_complete));
  }
  ~Interpreter() {
    if (handle_) {
        // ensure python uninitialization runs before we dlclose the library
        pImpl_.reset();
      // it segfaults its face off trying to unload, but it's not clear
      // if this is something we caused of if libtorch_python would also do the
      // same if it were opened/closed a lot...
      dlclose(handle_);
    }
  }
  Interpreter(Interpreter&& rhs)
      : library_name_(std::move(rhs.library_name_)),
        handle_(rhs.handle_),
        pImpl_(std::move(rhs.pImpl_)) {
    rhs.handle_ = nullptr;
  }

  Interpreter(const Interpreter&) = delete;
  Interpreter& operator=(const Interpreter&) = delete;
  Interpreter& operator=(Interpreter&&) = delete;
};

struct Package;

struct LoadBalancer {
  LoadBalancer(size_t n)
  : locks_(new uint64_t[8*n]), n_(n) {
    AT_ASSERT(n_ < 64, "max interpreters is 64 because I am lazy.");
    memset(locks_, 0, 8*n_*sizeof(uint64_t));
  }
  void setResourceLimit(size_t n) {
    n_ = n;
  }
  int acquire() {
    thread_local int last = 0;
    size_t minusers = SIZE_MAX;
    int min_idx = 0;
    for(size_t i = 0; i < n_; ++i, ++last) {
      if (last >= n_) {
        last = 0;
      }
      uint64_t prev = __atomic_fetch_add(&locks_[8*last], 1ULL, __ATOMIC_SEQ_CST);
      if (prev == 0) {
        // fast path, we found an interpreter with no users
        return last;
      } else {
        // slow path, we don't want to use this interpreter because it is being used by someone else.
        __atomic_fetch_sub(&locks_[8*last], 1ULL, __ATOMIC_SEQ_CST);
      }
      if (prev < minusers) {
        minusers = prev;
        min_idx = last;
      }
    }
    // we failed to find a completely free interpreter. heuristically use the one
    // with the least number of user (note that this may have changed since then, so this is only
    // a heuristic).
    __atomic_fetch_add(&locks_[8*min_idx], 1ULL, __ATOMIC_SEQ_CST);
    return min_idx;
  }
  void free(int where) {
    __atomic_fetch_sub(&locks_[8*where], 1ULL, __ATOMIC_SEQ_CST);
  }

private:
  uint64_t* locks_;
  size_t n_;
};


struct InterpreterManager {
  InterpreterManager(size_t n_interp = 2)
      : resources_(n_interp) {
    for (size_t i = 0; i < n_interp; ++i) {
      instances_.emplace_back();
      auto I = instances_.back().acquire_session();
      // make torch.version.interp be the interpreter id
      // can be used for balancing work across GPUs
      I.global("torch", "version").attr("__setattr__")({"interp", int64_t(i)});
    }
  }
  // get a free model, guarenteed that no other user of acquire_one has the same
  // model. It _is_ possible that other users will be using the interpreter.
  InterpreterSession acquire_one() {
    int where = resources_.acquire();
    return instances_[where].acquire_session([this, where] {
      resources_.free(where);
    });
  }

  // use to make sure something gets run on all interpreters, such as loading or
  // unloading a model eagerly
  at::ArrayRef<Interpreter> all_instances() {
    return instances_;
  }
  void debugLimitInterpreters(size_t N) {
    AT_ASSERT(N <= instances_.size());
    resources_.setResourceLimit(N);
  }
  Package load_package(const std::string& uri);
  InterpreterManager(const InterpreterManager&) = delete;
 private:
  friend struct Package;
  size_t next_object_id_ = 0;
  std::vector<Interpreter> instances_;
  LoadBalancer resources_;
};


struct MovableObject {
  InterpreterSession acquire_session(
      const Interpreter* on_this_interpreter = nullptr);
  at::IValue operator()(at::ArrayRef<at::IValue> args) {
    auto I = acquire_session();
    return I.self(args).toIValue();
  }
  void unload(const Interpreter* on_this_interpreter = nullptr);
  ~MovableObject() {
    unload();
  }

 private:
  friend struct Package;
  MovableObject(size_t object_id, PickledObject data, Package* package)
  : object_id_(object_id), data_(data), package_(package) {}
  int64_t object_id_;
  PickledObject data_;
  Package* package_;
};

struct Package {
  MovableObject load(std::function<PythonObject(InterpreterSession&)> ctor) {
    auto I = acquire_session();
    auto loaded = ctor(I);
    auto pickled = I.impl_->pickle(I.self, loaded);
    return MovableObject(package_manager_->next_object_id_++, std::move(pickled), this);
  }
  // shorthand for getting the object as a pickle resource in the package
  MovableObject load_pickle(const std::string& module, const std::string& file) {
    return load([&](InterpreterSession& I) {
      return I.self.attr("load_pickle")({module, file});
    });
  }

  InterpreterSession acquire_session() {
    auto I = package_manager_->acquire_one();
    I.self = I.impl_->create_or_get_package_importer_from_container_file(
        container_file_);
    return I;
  }

private:
 Package(
     const std::string& uri,
     InterpreterManager*
         pm) // or really any of the constructors to our zip file format
     : package_manager_(pm),
       container_file_(
           std::make_shared<caffe2::serialize::PyTorchStreamReader>(uri)) {}
 friend struct MovableObject;
 friend struct InterpreterManager;
 InterpreterManager* package_manager_;
 std::shared_ptr<caffe2::serialize::PyTorchStreamReader> container_file_;
};

} // namespace torch