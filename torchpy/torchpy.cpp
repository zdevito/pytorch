#include "torchpy.h"

namespace torch {

Package InterpreterManager::load_package(const std::string& uri) {
  return Package(uri, this);
}

InterpreterSession MovableObject::acquire_session(
    const Interpreter* on_this_interpreter) {
  InterpreterSession I = on_this_interpreter
      ? on_this_interpreter->acquire_session()
      : manager_->acquire_one();
  I.self = I.impl_->unpickle_or_get(object_id_, data_);
  return I;
}
void MovableObject::unload(const Interpreter* on_this_interpreter) {
  if (!on_this_interpreter) {
    for (auto& interp : manager_->all_instances()) {
      unload(&interp);
    }
    return;
  }

  InterpreterSession I = on_this_interpreter->acquire_session();
  I.impl_->unload(object_id_);
}

MovableObject InterpreterSession::create_movable(PythonObject obj) {
  TORCH_CHECK(manager_, "Can only create a movable object when the session was created from an interpreter that is part of a InterpreterManager");
  auto pickled = impl_->pickle(self, obj);
  return MovableObject(manager_->next_object_id_++, std::move(pickled), manager_);
}

} // namespace torch