#include <gtest/gtest.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);

  int rc = RUN_ALL_TESTS();

  return rc;
}
