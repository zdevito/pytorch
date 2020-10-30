#include <gtest/gtest.h>
#include <torchpy.h>
#include <iostream>
#include <string>
#include "torch/script.h"

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
