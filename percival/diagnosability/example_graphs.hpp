#pragma once

#include <memory>
#include <string>
#include <vector>

#include "percival/diagnosability/dataset.h"
#include "percival/diagnosability/typedefs.h"

namespace percival {
namespace example {

template <class FG>
DiagnosticGraphPtr<FG> ChainGraph(std::size_t num_failure_modes) {
  auto dgraph = std::make_shared<DiagnosticGraph<FG>>();
  std::vector<VarName> failure_mode_names;
  for (std::size_t i = 0; i < num_failure_modes; ++i) {
    auto fname = VarName("f" + std::to_string(i));
    dgraph->addFailureMode(fname, FailureModeType::OUTPUT);
    failure_mode_names.push_back(fname);
  }
  for (std::size_t i = 0; i < num_failure_modes - 1; ++i) {
    auto tname = VarName("t" + std::to_string(i));
    std::vector<VarName> scope = {failure_mode_names[i], failure_mode_names[i + 1]};
    dgraph->addTest(tname, tname, scope, &utils::PerfectTest);
  }
  return dgraph;
}

template <class FG>
DiagnosticGraphPtr<FG> OneDiagnosable() {
  auto dgraph = std::make_shared<DiagnosticGraph<FG>>();
  dgraph->addFailureMode("f1", FailureModeType::OUTPUT);
  dgraph->addFailureMode("f2", FailureModeType::OUTPUT);
  dgraph->addFailureMode("f3", FailureModeType::OUTPUT);
  dgraph->addTest("A", "A", {"f1", "f2"}, &utils::PerfectTest);
  dgraph->addTest("B", "B", {"f2", "f3"}, &utils::PerfectTest);
  dgraph->max_failure_cardinality(1);
  return dgraph;
}

template <class FG>
DiagnosticGraphPtr<FG> TwoDiagnosable() {
  auto dgraph = std::make_shared<DiagnosticGraph<FG>>();
  dgraph->addFailureMode("f1", FailureModeType::OUTPUT);
  dgraph->addFailureMode("f2", FailureModeType::OUTPUT);
  dgraph->addFailureMode("f3", FailureModeType::OUTPUT);
  dgraph->addFailureMode("f4", FailureModeType::OUTPUT);
  dgraph->addTest("A", "A", {"f1", "f3", "f4"}, &utils::PerfectTest);
  dgraph->addTest("B", "B", {"f2", "f4"}, &utils::PerfectTest);
  dgraph->addTest("C", "C", {"f1", "f3"}, &utils::PerfectTest);
  dgraph->addTest("D", "D", {"f1", "f2"}, &utils::PerfectTest);
  dgraph->addTest("E", "E", {"f2"}, &utils::PerfectTest);
  dgraph->addTest("F", "F", {"f3"}, &utils::PerfectTest);
  dgraph->max_failure_cardinality(2);
  return dgraph;
}

}  // namespace example
}  // namespace percival