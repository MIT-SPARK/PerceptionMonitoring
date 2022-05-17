#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>

#include "csv.hpp"
#include "gflags/gflags.h"
#include "percival/diagnosability/dataset.h"
#include "percival/diagnosability/diagnostic_graph.h"
#include "percival/diagnosability/factor_graph/grante.h"

DEFINE_bool(train, false, "Train the model");
DEFINE_bool(randomize, false, "Randomize energies");

using FactorGraph = percival::factor_graph::grante::FactorGraph;
namespace Inference = percival::factor_graph::grante::inference;

// clang-format off
// const std::string kDataset = "/home/antonap/sparklab/diagnosability/temporary/dummy.csv";
// const std::string kModel = "/home/antonap/sparklab/diagnosability/temporary/dummy.json";
// const std::string kParams = "/home/antonap/sparklab/diagnosability/temporary/trained.json";
// const std::string kTrainedModel = "/home/antonap/sparklab/diagnosability/temporary/trained.json";

const std::string kDataset = "/home/antonap/sparklab/diagnosability/temporary/test/test.csv";
const std::string kModel = "/home/antonap/sparklab/diagnosability/temporary/test/model.json";
const std::string kParams = "/home/antonap/sparklab/diagnosability/temporary/test/checkpoint.json";
const std::string kTrainedModel = "/home/antonap/sparklab/diagnosability/temporary/ignore.json";
// clang-format on

unsigned int HammingDistance(percival::SystemState a, percival::SystemState b) {
  unsigned int distance = 0;
  for (const auto &[key, value] : a) {
    if (value != b[key]) ++distance;
  }
  return distance;
}

int main(int argc, char *argv[]) {
  gflags::SetUsageMessage("Just an example");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  gflags::ShutDownCommandLineFlags();

  std::cout << "Loading model: " << kModel << std::endl;
  percival::DiagnosticGraph<FactorGraph> dgraph;
  dgraph.fromJson(kModel);
  if (FLAGS_randomize) dgraph.randomizeWeights();
  dgraph.bake();
  dgraph.summary();
  dgraph.print();

  dgraph.factor_graph()->training_options.ml_max_iterations = 0;
  dgraph.factor_graph()->training_options.ml_tolerance = 1e-4;
  dgraph.factor_graph()->training_options.cd_iterations = 0;
  dgraph.factor_graph()->training_options.ssvm_iterations = 100;
  dgraph.factor_graph()->training_options.ssvm_opt_method = "stochastic";
  dgraph.factor_graph()->training_options.ssvm_regularization = 1e3;

  std::cout << "Loading dataset: " << kDataset << std::endl;
  percival::Dataset dataset(kDataset, dgraph.failure_mode_names(), dgraph.test_names());
  std::cout << "Dataset size: " << dataset.size() << std::endl;
  if (FLAGS_train) {
    std::cout << "Training model..." << std::endl;
    dgraph.train(dataset);
    std::cout << "training completed." << std::endl;
    dgraph.toJson(kTrainedModel);
    dgraph.factor_graph()->save(kParams);
  } else {
    std::cout << "Skipping training..." << std::endl;
  }
  std::cout << "Running inference on Training dataset..." << std::endl;
  unsigned int distance = 0;
  for (const auto sample : dataset) {
    auto fi = dgraph.faultIdentification(sample.syndrome);
    distance += HammingDistance(sample.ground_truth, fi);
  }
  std::cout << "Inference on Training dataset completed." << std::endl;
  std::cout << "Average Hamming distance: " << distance / dataset.size() << std::endl;

  // ================

  std::cout << "Loading model: " << kParams << std::endl;
  percival::DiagnosticGraph<FactorGraph> dgraph_loaded;
  dgraph_loaded.fromJson(kModel);
  // dgraph_loaded.bake();
  dgraph_loaded.load(kParams);
  std::cout << "Done" << std::endl;

  std::cout << "Running inference on Training dataset..." << std::endl;
  distance = 0;
  for (const auto sample : dataset) {
    auto fi = dgraph_loaded.faultIdentification(sample.syndrome);
    distance += HammingDistance(sample.ground_truth, fi);
  }
  std::cout << "Inference on Training dataset completed." << std::endl;
  std::cout << "Average Hamming distance: " << distance / dataset.size() << std::endl;

  return 0;
}