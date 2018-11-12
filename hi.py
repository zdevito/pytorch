import torch

# @torch.jit.script
# def bar(a):
#     # type: (int) -> int
#     i = a
#     while(i < 10):
#         print(i)
#         i += 1
#     return i
#
# yepdss
# print(bar.graph.pretty_print())
# print(bar.graph)
#
#
# @torch.jit.script
# def baz(a, b):
#     print("Hiii!\08\n")
#     alist = torch.jit.annotate(List[float], [-3.0])
#     x = 4
#     for i in range(10):
#         print(i)
#         r = torch.jit._fork(bar, i)
#         x += torch.jit._wait(r)
#     return x, alist
#
# print(baz.graph.pretty_print())
#
# print(baz.graph)
#
#
# @torch.jit.script
# def foo(a):
#     # type: (Optional[int]) -> int
#     return 4
# dddddd
# @torch.jit.script
# def baz(b):
#     return foo(None)
#
#
# print(baz.graph)
#
#
a = torch.rand(4)
#


def foo(b):
    return b + a

r = torch.jit.trace(foo, torch.rand(4))

print(r.graph.pretty_print())
