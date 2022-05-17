#include <chrono>
#include <fstream>
#include <iostream>

#include "gflags/gflags.h"
#include "percival/diagnosability/diagnostic_graph.h"
#include "percival/diagnosability/factor_graph/grante.h"

DEFINE_string(model, "", "Path to diagnostic graph definition file [.json]");

static bool IsNonEmptyMessage(const char *flagname, const std::string &value) {
  return value[0] != '\0';
}
DEFINE_validator(model, &IsNonEmptyMessage);

using FactorGraph = percival::factor_graph::grante::FactorGraph;
namespace Inference = percival::factor_graph::grante::inference;

int main(int argc, char *argv[]) {
  gflags::SetUsageMessage("Train a Diagnostic Factor Graph");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  gflags::ShutDownCommandLineFlags();

  percival::DiagnosticGraph<FactorGraph> dgraph;
  std::cout << "Loading model: " << FLAGS_model << std::endl;
  {
    auto t_start = std::chrono::high_resolution_clock::now();
    dgraph.fromJson(FLAGS_model);
    dgraph.bake();
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "Model loaded in: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start)
                     .count()
              << "ms" << std::endl;
  }
  dgraph.summary();
  std::cout << "Number of parameters: " << dgraph.factor_graph()->numberOfParameters()
            << std::endl;
}