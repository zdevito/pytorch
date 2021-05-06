import lldb
# load into lldb instance with:
#   command script import debugger_script.py
def on_special_break():
    name_ptr, file_addr, file_size, load_bias = read_global_variable("__deploy_module_info")
    file_name = read_string(name_ptr)
    file_contents = read_mem(file_addr, file_size)
    temp_file = TempFile(with_hint=file_name)
    temp_file.write(file_contents)
    lldb.debugger.HandleCommand(f"target module add {temp_file.name}")
    lldb.debugger.HandleCommand(f"target module load -f {temp_file.name} {load_bias}")
    tell_debugger_not_to_stop()
target = lldb.debugger.GetSelectedTarget()
print(target)
bp = target.BreakpointCreateByRegex("__deploy_register_code")
bp.SetScriptCallbackBody("""\
process = frame.thread.GetProcess()
target = process.target
symbol_addr = frame.module.FindSymbol("__deploy_module_info").GetStartAddress()
info_addr = symbol_addr.GetLoadAddress(target)
e = lldb.SBError()
ptr_size = 8
str_addr = process.ReadPointerFromMemory(info_addr, e)
file_addr = process.ReadPointerFromMemory(info_addr + ptr_size, e)
file_size = process.ReadPointerFromMemory(info_addr + 2*ptr_size, e)
load_bias = process.ReadPointerFromMemory(info_addr + 3*ptr_size, e)
name = process.ReadCStringFromMemory(str_addr, 512, e)
r = process.ReadMemory(file_addr, file_size, e)
from tempfile import NamedTemporaryFile
from pathlib import Path
stem = Path(name).stem
with NamedTemporaryFile(prefix=stem, suffix='.so', delete=False) as tf:
    tf.write(r)
    print(tf.name)
    cmd1 = f"target modules add {tf.name}"
    # print(cmd1)
    lldb.debugger.HandleCommand(cmd1)
    cmd2 = f"target modules load -f {tf.name} -s {hex(load_bias)}"
    # print(cmd2)
    lldb.debugger.HandleCommand(cmd2)

return False
""")