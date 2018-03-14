#include "torch/csrc/jit/script/init.h"
#include "torch/csrc/jit/script/compiler.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"

namespace torch {
namespace jit {
namespace script {

using ResolutionCallback = std::function<py::function(std::string)>;

// The visibility attribute is to avoid a warning about storing a field in the
// struct that has a different visibility (from pybind) than the struct.
#ifdef _WIN32
#define VISIBILITY_HIDDEN
#else
#define VISIBILITY_HIDDEN __attribute__((visibility("hidden")))
#endif



static void ensureSizeMatches(SourceRange loc, size_t expected, size_t actual, const std::string& what) {
  if(expected != actual) {
    throw ErrorReport(loc) << "expected " << expected << " " << what << " but found " << actual;
  }
}

struct VISIBILITY_HIDDEN PythonValue : public SugaredValue {
  PythonValue(py::object self)
  : self(std::move(self)) {}

  // call it like a function, e.g. `outputs = this(inputs)`
  virtual std::vector<Value*> call(SourceRange loc, Method & m, at::ArrayRef<Value*> inputs, List<Attribute> attributes, size_t n_outputs) override {
    ensureSizeMatches(loc, 0, attributes.size(), "keyword arguments");
    // Release the function object so we can wrap it in a PythonOp
    Graph& g = *m.graph();
    py::object func = self;
    std::string cconv(inputs.size(), 't');
    Node* new_node = g.insertNode(g.createPythonOp(
      THPObjectPtr(func.release().ptr()), cconv, false, {}, {}, false));
    for(auto i : inputs)
      new_node->addInput(i);
    std::vector<Value*> outputs;
    for(size_t i = 0; i < n_outputs; ++i)
      outputs.push_back(new_node->addOutput());
    return outputs;
  }

  virtual std::string kind() const override {
    std::stringstream ss;
    ss << "python value'" << py::repr(self) << "'";
    return ss.str();
  }
private:
  py::object self;
};

Resolver pythonResolver(ResolutionCallback rcb) {
  return [=](const std::string& name) -> std::shared_ptr<SugaredValue> {
      AutoGIL ag;
      py::object obj = rcb(name);
      if(obj.is(py::none())) {
        return nullptr;
      }
      return std::make_shared<PythonValue>(obj);
  };
}

// defines how modules/methods behave inside the script subset.
// for now this does not have any interaction with python.
// in the future, we will add the ability to resolve `self.foo` to python
// {functions, modules, contants} so this SugaredValue is defined here
// anticipating we will eventually need to replace Module with a py::object
// holding the actual nn.Module class.

// defines how a method obtained from a module behaves in script
struct MethodValue : public SugaredValue {
  MethodValue(std::shared_ptr<Module> module, Method& method)
  : module(std::move(module)) //insurance that method stays alive
  , method(method) {}
  std::string kind() const override {
    return "method";
  }
  virtual std::vector<Value*> call(SourceRange loc, Method & caller, at::ArrayRef<Value*> inputs, List<Attribute> attributes, size_t n_outputs) override {
    if(attributes.size() != 0) {
      throw ErrorReport(loc) << "not yet implemented - calls to python functions using keyword arguments";
    }
    ensureSizeMatches(loc, caller.num_inputs(), inputs.size(), "inputs");
    auto outputs = caller.emit_call_to(method, inputs);
    ensureSizeMatches(loc, outputs.size(), n_outputs, "outputs");
    return outputs;
  }
private:
  std::shared_ptr<Module> module;
  Method& method;

};


struct ModuleValue : public SugaredValue {
  ModuleValue(std::shared_ptr<Module> module)
  : module(std::move(module)) {}

  virtual std::string kind() const override {
    return "module";
  }

  // select an attribute on it, e.g. `this.field`
  virtual std::shared_ptr<SugaredValue> attr(SourceRange loc, Method & m, const std::string& field) override {
    if(auto v = module->find_module(field)) {
      return std::make_shared<ModuleValue>(v->module);
    } else if(auto v = module->find_method(field)) {
      return std::make_shared<MethodValue>(module, *v);
    } else if(auto v = module->find_parameter(field)) {
      return std::make_shared<SimpleValue>(m.get_or_add_parameter(v->slot()));
    } else {
      throw ErrorReport(loc) << "module has no attribute '" << field << "'";
    }
  }
  // call module.forward
  virtual std::vector<Value*> call(SourceRange loc, Method & caller, at::ArrayRef<Value*> inputs, List<Attribute> attributes, size_t n_outputs) override {
    return attr(loc, caller, "forward")->call(loc, caller, inputs, attributes, n_outputs);
  }
private:
  std::shared_ptr<Module> module;
};

// TODO: dedup with other init

// we cannot use the default py:cast<autograd::Variable> because it currently
// unwraps the data tensor in the conversion process

variable_tensor_list createVariableTensorList(py::tuple tuple, size_t reserve_extra_space = 0) {
  variable_tensor_list result;
  result.reserve(tuple.size() + reserve_extra_space);
  for(auto e : tuple) {
    result.push_back(py::cast<autograd::Variable>(e));
  }
  return result;
}

py::object unpackVariableTensorList(std::vector<at::Tensor> outputs) {
  // if we don't tell pybind these are variables it chokes on the
  // conversion.
  // TODO: fix conversions to be sane and make sure this works.
  if(outputs.size() == 1) {
    return py::cast(static_cast<autograd::Variable&>(outputs[0]));
  } else {
    py::tuple tuple(outputs.size());
    for(size_t i = 0; i < outputs.size(); i++) {
      tuple[i] = py::cast(static_cast<autograd::Variable&>(outputs[i]));
    }
    return tuple;
  }
}

// This is a temporary constructor so that we can write python tests of
// the executor. It does not have most of the functionality of CompiledFunction
// such as being able to hold parameters...
std::shared_ptr<Graph> createGraphByTracing(
        py::function func,
        std::vector<tracer::TraceInput> trace_inputs,
        size_t num_func_inputs) {
  auto enter_info = tracer::enter(std::move(trace_inputs), 1);
  py::tuple py_inputs(num_func_inputs);
  for(size_t i = 0; i < num_func_inputs; ++i) {
    py_inputs[i] = py::cast(enter_info.second[i]);
  }
  auto out = func(*py_inputs);
  std::vector<autograd::Variable> outputs;
  if(PyTuple_Check(out.ptr())) {
    outputs = py::cast<std::vector<autograd::Variable>>(out);
  } else {
    outputs.push_back(py::cast<autograd::Variable>(out));
  }
  tracer::exit(outputs);
  auto graph = enter_info.first->graph;
  EliminateDeadCode(graph);
  return graph;
}

static void gatherStateDict(std::vector<at::Tensor*> & values, const Module & m) {
  for(auto & params : m.get_parameters()) {
    values.push_back(params.slot());
  }
  for(const auto & sub : m.get_modules()) {
    gatherStateDict(values, *sub.module);
  }
}

void initJitScriptBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  // torch.jit.ScriptModule is a subclass of this C++ object.
  // Methods here are prefixed with _ since they should not be
  // public.
  py::class_<Module, std::shared_ptr<Module>>(m, "ScriptModule")
      .def(py::init<bool>())
      .def(
          "_define",
          [](Module& m,
             const std::string& script,
             ResolutionCallback rcb, bool has_self) {
            auto self = has_self ? std::make_shared<ModuleValue>(m.shared_from_this()) : nullptr;
            return defineMethodsInModule(m, script, pythonResolver(rcb), self);
          })
      .def("_create_method", [](Module& m, Def def, ResolutionCallback rcb) {
        defineMethodsInModule(
          m,
          { def },
          pythonResolver(rcb),
          std::make_shared<ModuleValue>(m.shared_from_this()));
      })
      .def("_get_method",
      [](Module& self, const std::string& name) -> const Method& {
        return self.get_method(name);
      }, py::return_value_policy::reference_internal)
      .def("_register_parameter", &Module::register_parameter)
      .def("_register_module", &Module::register_module)
      .def("_set_parameter", &Module::set_parameter)
      .def("_get_parameter", &Module::get_parameter)
      .def("_get_module", &Module::get_module)
      .def("_get_modules", [](Module& self) -> py::tuple {
        auto & modules = self.get_modules();
        py::tuple result(modules.size());
        for(size_t i = 0; i < modules.size(); ++i) {
          auto & nm = modules[i];
          py::tuple pair(2);
          pair[0] = nm.name;
          pair[1] = nm.module;
          result[i] = pair;
        }
        return result;
      })
      .def("_get_parameters", [](Module& self) -> py::tuple {
        auto & parameters = self.get_parameters();
        py::tuple result(parameters.size());
        for(size_t i = 0; i < parameters.size(); ++i) {
          auto & p = parameters[i];
          py::tuple r(3);
          r[0] = p.name;
          r[1] = static_cast<const autograd::Variable&>(*p.slot());
          r[2] = p.is_buffer;
          result[i] = r;
        }
        return result;
      })
      .def("_has_parameter", [](Module& self, const std::string& name) {
        if(auto r = self.find_parameter(name)) {
          return !r->is_buffer;
        }
        return false;
      })
      .def("_has_buffer", [](Module& self, const std::string& name) {
        if(auto r = self.find_parameter(name)) {
          return r->is_buffer;
        }
        return false;
      })
      .def("_has_module", [](Module& self, const std::string& name) {
        return bool(self.find_module(name));
      })
      .def("_has_method", [](Module& self, const std::string& name) {
        return bool(self.find_method(name));
      })
      .def("_create_method_from_trace", [](
        Module& self,
        const std::string& name,
        py::function func,
        std::vector<tracer::TraceInput> inputs) {
          size_t num_inputs = inputs.size();
          std::vector<at::Tensor*> parameters;
          gatherStateDict(parameters, self);
          for(at::Tensor* param : parameters) {
            inputs.push_back(static_cast<autograd::Variable&>(*param));
          }
          auto graph = createGraphByTracing(func, std::move(inputs), num_inputs);
          self.create_method(name, std::move(graph), std::move(parameters));
      });

  py::class_<Method>(m, "ScriptMethod")
    .def("graph", [&](Method& self) {
      return self.graph();
    })
    .def("__call__", [](Method& m, py::args args) -> py::object {
      auto inputs = createVariableTensorList(args);
      auto outputs = m.run(std::move(inputs));
      return unpackVariableTensorList(std::move(outputs));
    });

  m.def("_jit_script_compile", [](Def def, ResolutionCallback rcb) {
    return compileFunction(def, pythonResolver(rcb));
  });

}

} // namespace script
} // namespace jit
} // namespace torch
