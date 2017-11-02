#include <Python.h>
#include <iostream>
#ifdef WITH_CUDA
#include "torch/csrc/jit/fusion_compiler.h"
#endif
#include "torch/csrc/jit/code_template.h"
#include "torch/csrc/assertions.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/attributes.h"
#include "torch/csrc/jit/interned_strings.h"
#include <vector>

#include "torch/csrc/jit/generated/aten_dispatch.h"
#include "torch/csrc/jit/benchmark_common.h"


namespace torch { namespace jit {

// The interpreter run Graphs with Tensor inputs and Tensor outputs
// a separate component in the autograd handles unwrapping and wrapping
// variable objects for use in the interpreter.


// We need some lists for inputs and outputs. To keep all the memory
// contiguous we allocate a single vector and use offsets into the vector
// which are stored in the RegList struct
// start is an offset into int_data of Function if this list is integers
// and bool_data of Function if this list is booleans (only free_flags)
struct RegList {
  int start;
  int size;
};

struct UseList {
  // values to be used
  RegList values;
  // boolean flags indicating whether to free the Tensor after this use
  RegList free_flags;
};

using Callback = std::function<void(const std::vector<at::Tensor> &, std::vector<at::Tensor>&)>;
// one instruction plus meta-data
struct Instruction {
  Callback callback;
  UseList inputs;
  RegList outputs;
};

struct Stage {
  RegList inputs; // inputs to define for the stage
  UseList outputs; // values consumed by the return
  std::vector<Instruction> instructions;
};

// pre-processing that happens once per graph
struct Function {
  Function(std::shared_ptr<Graph> & graph)
  : graph(graph) {
    int cur_stage = -1;
    int input_pos = 0;
    int output_pos = 0;
    // step 1: encode all operators and stages into registers and fill in
    // input/output lists
    for(auto node : graph->nodes()) {
      if(node->kind() == kSelect)
        continue;
      insertStagesTo(cur_stage, node->stage(), input_pos, output_pos);
      cur_stage = node->stage();
      stages.back().instructions.emplace_back();
      auto & inst = stages.back().instructions.back();
      intListBegin(inst.inputs.values);
      for(auto input : node->inputs()) {
        intListInsert(inst.inputs.values, getOrAllocateRegister(input, true));
      }
      intListBegin(inst.outputs);
      for(auto output : node->outputs()) {
        intListInsert(inst.outputs, getOrAllocateRegister(output));
      }
      inst.callback = std::move(getTensorOp(node).op);
    }
    insertStagesTo(cur_stage, graph->stage(), input_pos, output_pos);

    // step 2: the last time we use a register  we want to mark its free_flag
    // so we clean it up
    // this is done with a backward scan where we mark the first time we see it
    std::unordered_set<int> seen_registers;
    auto scanUses = [&](UseList & u) {
      boolListBegin(u.free_flags);
      for(size_t i = 0; i < u.values.size; i++) {
        int reg = Int(u.values,i);
        boolListInsert(u.free_flags, seen_registers.count(reg) == 0);
        seen_registers.insert(reg);
      }
    };
    for(auto sit = stages.rbegin(); sit != stages.rend(); sit++) {
      scanUses(sit->outputs);
      for(auto iit = sit->instructions.rbegin(); iit != sit->instructions.rend(); iit++) {
        scanUses(iit->inputs);
      }
    }
  }
private:
  void insertStagesTo(int cur_stage, int goal_stage, int & input_pos, int & output_pos) {
    while(cur_stage < goal_stage) {
      cur_stage++;
      stages.emplace_back();
      auto & stage = stages.back();
      intListBegin(stage.inputs);
      for(;input_pos < graph->inputs().size(); input_pos++) {
        auto input = graph->inputs()[input_pos];
        if(input->stage() > cur_stage)
          break;
        // unused inputs are given a false register -1 so that we never hold a
        // reference to the tensor data, otherwise we would fail to clean them
        // up since they do not have a last use at which to free them
        int reg = input->uses().size() > 0 ? getOrAllocateRegister(input) : -1;
        intListInsert(stage.inputs, reg);
      }
      intListBegin(stage.outputs.values);
      for(;output_pos < graph->outputs().size(); output_pos++) {
        auto output = graph->outputs()[output_pos];
        if(output->stage() > cur_stage)
          break;
        intListInsert(stage.outputs.values, getOrAllocateRegister(output));
      }
    }
  }
  // helpers to build/access RegList objects
  int Int(RegList & list, int i) {
    return int_data[list.start + i];
  }
  void intListBegin(RegList & list) {
    list.start = int_data.size();
    list.size = 0;
  }
  void intListInsert(RegList & list, int value) {
    int_data.push_back(value);
    list.size++;
  }
  void boolListBegin(RegList & list) {
    list.start = bool_data.size();
    list.size = 0;
  }
  void boolListInsert(RegList & list, int value) {
    bool_data.push_back(value);
    list.size++;
  }

  int getOrAllocateRegister(Node * n, bool required = false) {
    size_t u = n->unique();
    if(unique_to_reg.count(u) > 0)
      return unique_to_reg[u];
    JIT_ASSERT(!required);
    int r = register_size++;
    unique_to_reg[u] = r;
    return r;
  }
  std::shared_ptr<Graph> graph;
  std::unordered_map<size_t, int> unique_to_reg; // map from unique of nodes to register in register table

  friend struct Interpreter;
  std::vector<Stage> stages;
  int register_size = 0;

  // all memory ArrayRef<int> are slices of this, to make sure
  // the interpreter is mostly linearly scanning through memory
  std::vector<int> int_data;
  std::vector<bool> bool_data;
};

// Interpreter state that is held across stages and used to compute a Function
struct Interpreter {
  Interpreter(const std::shared_ptr<Function> & function)
  : function(function),
    int_data(function->int_data.data()),
    bool_data(function->bool_data),
    registers(function->register_size) {

  }
  void runOneStage(
    const std::vector<at::Tensor> & inputs,
    std::vector<at::Tensor> & outputs) {
      //std::cout << "running stage: " << current_stage << "\n";
      JIT_ASSERT(current_stage < function->stages.size());
      auto & stage = function->stages[current_stage++];
      JIT_ASSERT(inputs.size() == stage.inputs.size);
      for(size_t i = 0; i < stage.inputs.size; i++) {
        int reg = Int(stage.inputs,i);
        if(reg >= 0) { // otherwise this input is dead, and we do not store it to avoid holding the reference
          registers[reg] = inputs[i];
        }
        //std::cout << "registers[" << reg << "] = inputs[" << i << "](" << inputs[i].defined() << ")\n";
      }
      for(auto & inst : stage.instructions) {
        loadTensorsFromRegisters(inst.inputs, input_buffer);
        inst.callback(input_buffer, output_buffer);
        for(size_t i = 0; i < inst.outputs.size; i++) {
          int reg = Int(inst.outputs,i);
          registers[reg] = std::move(output_buffer[i]);
          //std::cout << "registers[" << reg << "] = outputs[" << i << "](" << output_buffer[i].defined() << ")\n";
        }
        output_buffer.clear();
        input_buffer.clear();
      }
      outputs.clear();
      loadTensorsFromRegisters(stage.outputs, outputs);
  }
private:
  int Int(const RegList & list, int i) {
    return int_data[list.start + i];
  };
  bool Bool(const RegList & list, int i) {
    return bool_data[list.start + i];
  }
  void loadTensorsFromRegisters(const UseList & uses, std::vector<at::Tensor> & outputs) {
    for(size_t i = 0; i < uses.values.size; i++) {
      int reg = Int(uses.values,i);
      auto & value = registers[reg];
      //std::cout << "inputs[" << i << "] = registers[" << reg << "] (" << value.defined() << ")";
      if(Bool(uses.free_flags,i)) {
        //std::cout << " and FREED";
        outputs.push_back(std::move(value));
      } else {
        outputs.push_back(value);
      }
      //std::cout << "\n";
    }
  }
  size_t current_stage = 0;
  std::shared_ptr<Function> function;
  // these are just copies of function to prevent indirections in intepreter
  int * int_data;
  const std::vector<bool> & bool_data;


  // this holds all the tensors for this interpreter run
  // we don't both minimizing the size of this vector, since the extra
  // memory used by the pointers in this will be small
  // instead we are very aggresive about releasing tensors when they become dead
  // to make sure memory management happens efficiently.
  std::vector<at::Tensor> registers;

  // single buffer for input calls to ATen functions, so that we do not reallocate
  std::vector<at::Tensor> input_buffer;
  // also to prevent allocations
  std::vector<at::Tensor> output_buffer;
};

at::Tensor t_use(at::Tensor x) {
  return x;
}
at::Tensor t_def(at::Tensor x) {
  return x.t();
}

// given the difference of output vs expected tensor, check whether the
// difference is within a relative tolerance range. This is a standard way of
// matching tensor values upto certain precision
bool checkRtol(const at::Tensor& diff, const std::vector<at::Tensor> inputs) {
  double maxValue = 0.0;
  for (auto& tensor : inputs) {
    maxValue = fmax(tensor.abs().max().toCFloat(), maxValue);
  }
  return diff.abs().max().toCFloat() < 2e-6 * maxValue;
}
bool almostEqual(const at::Tensor & a, const at::Tensor & b) {
  return checkRtol(a - b,{a, b});
}

std::pair<at::Tensor, at::Tensor>
lstm(at::Tensor input,
      at::Tensor hx,
      at::Tensor cx,
      at::Tensor w_ih,
      at::Tensor w_hh) {
  auto gates = input.mm(t_use(w_ih)) + hx.mm(t_use(w_hh));

  auto chunked_gates = gates.chunk(4, 1);
  auto ingate     = chunked_gates[0];
  auto forgetgate = chunked_gates[1];
  auto cellgate = chunked_gates[2];
  auto outgate    = chunked_gates[3];

  ingate = ingate.sigmoid();
  outgate = outgate.sigmoid();
  cellgate = cellgate.tanh();
  forgetgate = forgetgate.sigmoid();

  auto cy = (forgetgate * cx) + (ingate * cellgate);
  auto hy = outgate * cy.tanh();

  return {hy, cy};
}

Symbol sym(const char * str) {
  return stringToSymbol(str);
}

Node * node(Graph& graph, const char * n, ArrayRef<Node*> inputs) {
  return graph.appendNode(graph.create(sym(n),inputs));
}

Node * add(Graph & g, Node * a, Node * b) {
  auto r = node(g, "add", {a,b});
  r->t_(sym("alpha"), at::Scalar(1).toTensor());
  return r;
}

std::shared_ptr<Graph> build_lstm() {
  auto r = std::make_shared<Graph>();
  auto & g = *r;
  Node * input = g.addInput();
  Node * hx = g.addInput();
  Node * cx = g.addInput();
  Node * w_ih = g.addInput();
  Node * w_hh = g.addInput();


  auto gates = add(g, node(g,"mm",{ input, w_ih }), node(g, "mm", {hx, w_hh}));
  auto chunked_gates = node(g, "chunk", { gates });
  chunked_gates->i_(sym("chunks"), 4);
  chunked_gates->i_(sym("dim"), 1);
  auto ingate = g.appendNode(g.createSelect(chunked_gates, 0));
  auto forgetgate = g.appendNode(g.createSelect(chunked_gates, 1));
  auto cellgate = g.appendNode(g.createSelect(chunked_gates, 2));
  auto outgate = g.appendNode(g.createSelect(chunked_gates, 3));
  ingate = node(g,"sigmoid",{ingate});
  outgate = node(g,"sigmoid",{outgate});
  cellgate = node(g,"tanh",{cellgate});
  forgetgate = node(g,"sigmoid",{forgetgate});

  auto cy = add(g, node(g,"mul", {forgetgate, cx}) , node(g, "mul", {ingate, cellgate}));
  auto hy = node(g, "mul", {outgate, node(g, "tanh", {cy})});

  g.registerOutput(hy);
  g.registerOutput(cy);
  g.lint();

  return r;
}


int run_bench(int input_size) {

  constexpr unsigned int cpu = 0, gpu = 0;

  cpu_pin(cpu);
  check_cpu_governor(cpu);
  check_gpu_applications_clock(gpu);

  constexpr int batch_size = 1;
  //constexpr int input_size = ;
  int hidden_size = 2*input_size;

  constexpr int fast = 0;

  constexpr int seq_len = fast ? 3 : 512;
  constexpr int warmup = fast ? 2 : 10;
  constexpr int loops  = fast ? 3 : 20;

  auto input = at::CUDA(at::kFloat).randn({seq_len, batch_size, input_size});
  auto hx    = at::CUDA(at::kFloat).randn({batch_size, hidden_size});
  auto cx    = at::CUDA(at::kFloat).randn({batch_size, hidden_size});
  auto w_ih  = t_def(at::CUDA(at::kFloat).randn({4 * hidden_size, input_size}));
  auto w_hh  = t_def(at::CUDA(at::kFloat).randn({4 * hidden_size, hidden_size}));


  auto run_it = [&](const char * name, std::function<void(void)> body) {
    // Possible experiment:
    // Create a stream that is default nonblocking
    // (don't use the default stream because shenanigans)

    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));

    for (int i = 0; i < warmup + loops; i++) {
      CUDA_CHECK(cudaEventRecord(start, 0));
      auto start_cpu_ns = getTime();
      body();
      /*
      for (int j = 0; j < 300000; j++) {
        __asm__("");
      }
      */
      auto end_cpu_ns = getTime();
      CUDA_CHECK(cudaEventRecord(end, 0));
      CUDA_CHECK(cudaDeviceSynchronize());
      float gpu_msecs;
      cudaEventElapsedTime(&gpu_msecs, start, end);
      if(i + 1 == warmup + loops) {
        std::cout << name << " " << input_size << " " << gpu_msecs * 1000.0 / seq_len << " " << (end_cpu_ns-start_cpu_ns)/1000.0 / seq_len << "\n";
        //print_result_usecs(name, i, gpu_msecs * 1000, (end_cpu_ns-start_cpu_ns)/1000.0, seq_len);
      }
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
  };


  auto lstm_g = build_lstm();
  auto lstm_function = std::make_shared<Function>(lstm_g);
  std::vector<at::Tensor> outputs;
  Interpreter lstm_interp(lstm_function);
  lstm_interp.runOneStage({input[0], hx, cx, w_ih, w_hh}, outputs);
  std::tie(hx, cx) = lstm(input[0], hx, cx, w_ih, w_hh);

  //std::cout << almostEqual(outputs[0],hx) << "\n";
  JIT_ASSERT(almostEqual(outputs[0],hx));
  JIT_ASSERT(almostEqual(outputs[1],cx));


  run_it("lstm",[&]() {
    for (int j = 0; j < seq_len; j++) {
      std::tie(hx, cx) = lstm(input[j], hx, cx, w_ih, w_hh);
    }
  });

  run_it("lstm_interp",[&]() {
    outputs[0] = hx;
    outputs[1] = cx;
    for (int j = 0; j < seq_len; j++) {
      Interpreter lstm_interp(lstm_function);
      lstm_interp.runOneStage({input[j], outputs[0], outputs[1], w_ih, w_hh}, outputs);
    }
  });





  return 0;
}


}}


namespace torch { namespace jit {

template<typename T>
static std::ostream & operator<<(std::ostream & out, const std::vector<T> & list) {
  size_t i = 0;
  out << "{";
  for(auto && e : list) {
    if(i++ > 0)
      out << ", ";
    out << e;
  }
  out << "}";
  return out;
}
static auto ct = CodeTemplate(R"(
int foo($args) {

    $bar
        $bar
    $a+$b
}
int commatest(int a${,stuff})
int notest(int a${,empty,})
)");
static auto ct_expect = R"(
int foo(hi, 8) {

    what
    on many
    lines...
    7
        what
        on many
        lines...
        7
    3+4
}
int commatest(int a, things..., others)
int notest(int a)
)";

static void codeTemplateTest() {
  {
    TemplateEnv e;
    e.s("hi","foo");
    e.v("what",{"is","this"});
    TemplateEnv c(e);
    c.s("hi","foo2");
    JIT_ASSERT(e.s("hi") == "foo");
    JIT_ASSERT(c.s("hi") == "foo2");
    JIT_ASSERT(e.v("what")[0] == "is");
  }

  {
    TemplateEnv e;
    e.v("args",{"hi","8"});
    e.v("bar",{"what\non many\nlines...","7"});
    e.s("a","3");
    e.s("b","4");
    e.v("stuff",{"things...","others"});
    e.v("empty",{});
    auto s = ct.format(e);
    //std::cout << "'" << s << "'\n";
    //std::cout << "'" << ct_expect << "'\n";
    JIT_ASSERT(s == ct_expect);
  }
}

#ifdef WITH_CUDA
Node * appendNewNode(NodeKind kind, Graph& graph, ArrayRef<Node*> inputs) {
  return graph.appendNode(graph.create(kind,inputs));
}

static void fusionTests() {
  FusionCompiler comp;
  cudaFree(0);

  auto testSimple = [&] {
    Graph graph;
    Node * i0 = graph.addInput();
    Node * i1 = graph.addInput();
    auto o0 = appendNewNode(kmul,graph,{i0, i1});
    graph.registerOutput(o0);
    auto a = at::CUDA(at::kFloat).rand({3,4});
    auto b = at::CUDA(at::kFloat).rand({4,3}).transpose(0,1);
    auto o = at::CUDA(at::kFloat).zeros({3,4});
    comp.debugLaunchGraph(graph, {a,b}, {o});
    auto o2 = a*b;
    float max_diff = (o2 - o).abs().max().toCDouble();
    //std::cout << "max diff: " << max_diff << "\n";
    JIT_ASSERT(max_diff == 0);
  };
  testSimple();

  auto testOne = [&](int ti, int tj, int toi, int toj) {

    Graph graph;

    Node * i0 = graph.addInput();
    Node * i1 = graph.addInput();
    Node * i2 = graph.addInput();
    Node * i3 = graph.addInput();
    Node * i4 = graph.addInput();

    auto p22 = appendNewNode(ksigmoid,graph,{i4});
    auto p20 = appendNewNode(ksigmoid,graph,{i3});
    auto p18 = appendNewNode(ktanh,graph,{i2});
    auto p16 = appendNewNode(ksigmoid,graph,{i1});
    auto p14 = appendNewNode(kmul,graph,{p20, i0});
    auto p11 = appendNewNode(kmul,graph,{p22, p18});
    auto o1 = appendNewNode(kadd,graph,{p14, p11});
    o1->t_(kalpha, at::Scalar(1).toTensor());
    auto p5 = appendNewNode(ktanh,graph,{o1});
    auto o0 = appendNewNode(kmul,graph,{p16, p5});

    graph.registerOutput(o0);
    graph.registerOutput(o1);

    graph.lint();

    std::vector<at::Tensor> inputs;
    std::vector<at::Tensor> outputs;
    // We want to generate input/output tensors with dimension 128x128x32, but
    // with different internal strides.  To do this, we generate a tensor
    // with the "wrong" dimensions, and then use transpose to get an appropriately
    // sized view.
    for(size_t i = 0; i < graph.inputs().size(); i++) {
      std::vector<int64_t> dims = {128, 128, 32};
      std::swap(dims[ti],dims[tj]);
      inputs.push_back(at::CUDA(at::kFloat).rand(dims).transpose(ti, tj));
    }
    for(size_t i = 0; i < graph.outputs().size(); i++) {
      std::vector<int64_t> dims = {128, 128, 32};
      std::swap(dims[toi],dims[toj]);
      outputs.push_back(at::CUDA(at::kFloat).zeros(dims).transpose(toi,toj));
    }

    auto t22 = inputs[4].sigmoid();
    auto t20 = inputs[3].sigmoid();
    auto t18 = inputs[2].tanh();
    auto t16 = inputs[1].sigmoid();
    auto t14 = t20*inputs[0];
    auto t11 = t22*t18;
    auto out1 = t14+t11;
    auto t5 = out1.tanh();
    auto out0 = t16*t5;


    //auto out0 = inputs[0]*inputs[1];
    comp.debugLaunchGraph(graph, inputs, outputs);
    JIT_ASSERT(out0.is_same_size(outputs.front()));
    float max_diff = (outputs.front() - out0).abs().max().toCDouble();
    JIT_ASSERT(max_diff < 1e-6);

  };
  testOne(0,0,0,0);
  testOne(0,1,0,0);
  testOne(1,2,0,0);
  testOne(0,2,0,0);

  testOne(0,0,0,1);
  testOne(0,1,1,2);
  testOne(1,2,0,2);



  auto testConcat = [&](int dim) {
    Graph graph;
    Node * i0 = graph.addInput();
    Node * i1 = graph.addInput();
    auto o0 = appendNewNode(kmul,graph,{i0, i1});
    graph.registerOutput(o0);
    graph.registerOutput(appendNewNode(kcat, graph, {i0,o0})->i_(kdim, dim));
    auto a = at::CUDA(at::kFloat).rand({3,4,5});
    auto b = at::CUDA(at::kFloat).rand({4,3,5}).transpose(0,1);
    auto o = at::CUDA(at::kFloat).zeros({3,4,5});

    auto o_r = a*b;
    auto o2_r = at::cat({a, o_r}, dim);
    auto o2 = at::CUDA(at::kFloat).zeros(o2_r.sizes());
    comp.debugLaunchGraph(graph, {a,b}, {o, o2});

    float max_diff = (o_r - o).abs().max().toCDouble();
    JIT_ASSERT(max_diff == 0);
    float max_diff2 = (o2_r - o2).abs().max().toCDouble();
    JIT_ASSERT(max_diff2 == 0);
  };
  testConcat(0);
  testConcat(1);
  testConcat(2);
}

#else //WITH_CUDA
void fusionTests() {}
#endif
struct Attr : public Attributes<Attr> {
};
void attributesTest() {
  auto one = kParam;
  auto two = kReturn;
  auto three = kConstant;
  auto four = kSlice;
  Attr attr;
  attr.f_(one,3.4)->i_(two,5)->s_(three,"what");
  JIT_ASSERT(attr.f(one) == 3.4);
  JIT_ASSERT(attr.s(three) == "what");
  JIT_ASSERT(attr.i(two) == 5);
  attr.s_(one,"no");
  JIT_ASSERT(attr.s(one) == "no");
  JIT_ASSERT(attr.hasAttribute(three));
  JIT_ASSERT(!attr.hasAttribute(four));
  attr.ss_(two, {"hi", "now"});
  JIT_ASSERT(attr.ss(two).at(1) == "now");

  Attr attr2;
  attr2.copyAttributes(attr);
  JIT_ASSERT(attr2.s(one) == "no");
  attr2.f_(one,5);
  JIT_ASSERT(attr.s(one) == "no");
  JIT_ASSERT(attr2.f(one) == 5);
}

void internedStringsTests () {

  JIT_ASSERT(kParam == stringToSymbol("Param"));
  JIT_ASSERT(kReturn == stringToSymbol("Return"));
  JIT_ASSERT(symbolToString(kReturn) == std::string("Return"));
  size_t symstart = stringToSymbol("__NEW_SYMBOL");
  JIT_ASSERT(stringToSymbol("What") == symstart+1);
  JIT_ASSERT(stringToSymbol("What2") == symstart+2);
  JIT_ASSERT(stringToSymbol("What") == symstart+1);
  JIT_ASSERT(stringToSymbol("What2") == symstart+2);
  JIT_ASSERT(symbolToString(symstart+2) == std::string("What2"));
}


void runJITCPPTests() {
  for(auto i  : {1, 2, 4, 8, 16, 32, 64, 128, 256 }) {
    run_bench(i);
  }
  printf("DONE\n");
  exit(0);
  codeTemplateTest();
  fusionTests();
  attributesTest();
  internedStringsTests();
}

}}
