#include <iostream>

#include "percival/diagnosability/diagnostic_graph.h"
#include "percival/diagnosability/example_graphs.hpp"
#include "percival/diagnosability/factor_graph/grante.h"
#include "percival/diagnosability/factor_graph/opengm.h"

// using FactorGraph = percival::factor_graph::opengm::FactorGraph;
using FactorGraph = percival::factor_graph::grante::FactorGraph;
// using InferenceAlgorithm = percival::factor_graph::grante::inference::Algorithm;
namespace Inference = percival::factor_graph::grante::inference;

void print(const percival::SystemState &state) {
  for (const auto &[name, value] : state) {
    auto s = (value == percival::FailureModeState::ACTIVE) ? "ACTIVE" : "INACTIVE";
    std::cout << name << ": " << s << std::endl;
  }
  std::cout << std::endl;
}

void print(const percival::Syndrome &syn) {
  for (const auto &[name, value] : syn) {
    auto s = (value == percival::TestOutcome::PASS) ? "PASS" : "FAIL";
    std::cout << name << ": " << s << std::endl;
  }
  std::cout << std::endl;
}

int main() {
  // Create the example factor graph.
  auto dgraph = percival::example::OneDiagnosable<FactorGraph>();
  dgraph->summary();
  dgraph->bake();

  // Change the defailt inference options
  auto fg = dgraph->factor_graph();
  fg->inference_options.algorithm = Inference::Algorithm::kBeliefPropagation;
  fg->inference_options =
      Inference::DefaultInferenceOptions(fg->inference_options.algorithm);
  fg->inference_options.parameters.standard.max_iterations = 0;
  fg->inference_options.verbose = false;

  // Perform fault identification
  {
    percival::Syndrome syn;
    syn["A"] = percival::TestOutcome::PASS;
    syn["B"] = percival::TestOutcome::PASS;
    auto state = dgraph->faultIdentification(syn);

    std::cout << std::endl << "Fault identification result:" << std::endl;
    std::cout << "SYNDROME:" << std::endl;
    print(syn);
    std::cout << "FAULTS STATE:" << std::endl;
    print(state);
  }
  {
    percival::Syndrome syn;
    syn["A"] = percival::TestOutcome::PASS;
    syn["B"] = percival::TestOutcome::FAIL;
    auto state = dgraph->faultIdentification(syn);

    std::cout << std::endl << "Fault identification result:" << std::endl;
    std::cout << "SYNDROME:" << std::endl;
    print(syn);
    std::cout << "FAULTS STATE:" << std::endl;
    print(state);
  }
  {
    percival::Syndrome syn;
    syn["A"] = percival::TestOutcome::FAIL;
    syn["B"] = percival::TestOutcome::PASS;
    auto state = dgraph->faultIdentification(syn);

    std::cout << std::endl << "Fault identification result:" << std::endl;
    std::cout << "SYNDROME:" << std::endl;
    print(syn);
    std::cout << "FAULTS STATE:" << std::endl;
    print(state);
  }
  {
    percival::Syndrome syn;
    syn["A"] = percival::TestOutcome::FAIL;
    syn["B"] = percival::TestOutcome::FAIL;
    auto state = dgraph->faultIdentification(syn);

    std::cout << std::endl << "Fault identification result:" << std::endl;
    std::cout << "SYNDROME:" << std::endl;
    print(syn);
    std::cout << "FAULTS STATE:" << std::endl;
    print(state);
  }

  return 0;
}