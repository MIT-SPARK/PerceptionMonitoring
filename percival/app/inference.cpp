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
#include "progressbar.hpp"

DEFINE_bool(verbose, false, "Display verbose output");
DEFINE_string(model, "", "Path to diagnostic graph definition file [.json]");
DEFINE_string(params, "", "Path to parameters file [.dat]");
DEFINE_string(dataset, "", "Path to dataset [.csv]");
DEFINE_string(output, "", "Output folder");
// DEFINE_string(inference, "simulated_annealing", "Inference algorithm");
// DEFINE_double(tolerance, {}, "Inference algorithm tolerance");
// DEFINE_uint32(max_iterations, {}, "Inference algorithm maximum number of iteration");

static bool IsNonEmptyMessage(const char *flagname, const std::string &value) {
  return value[0] != '\0';
}
DEFINE_validator(model, &IsNonEmptyMessage);
DEFINE_validator(output, &IsNonEmptyMessage);
DEFINE_validator(dataset, &IsNonEmptyMessage);
// DEFINE_validator(inference, &IsNonEmptyMessage);
// DEFINE_validator(tolerance, [](const char *flagname, double value) { return value >
// 0; }); DEFINE_validator(max_iterations, [](const char *flagname, uint32_t value) {
// return value >= 0; });

using FactorGraph = percival::factor_graph::grante::FactorGraph;
namespace Inference = percival::factor_graph::grante::inference;
// using FactorGraph = percival::factor_graph::opengm::FactorGraph;

const std::string kInferenceOutputFilename = "inference.csv";
const std::string kTimingOutputFilename = "timing.csv";

int main(int argc, char *argv[]) {
  gflags::SetUsageMessage("Perform inference using a Diagnostic Factor Graph");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  gflags::ShutDownCommandLineFlags();

  std::cout << "Loading model: " << FLAGS_model << std::endl;
  percival::DiagnosticGraph<FactorGraph> dgraph;
  dgraph.fromJson(FLAGS_model);
  if (!FLAGS_params.empty()) {
    std::cout << "Loading params: " << FLAGS_params << std::endl;
    dgraph.load(FLAGS_params);
  } else {
    dgraph.bake();
  }
  if (FLAGS_verbose) {
    dgraph.summary();
    dgraph.print();
  }

  std::cout << "Loading dataset: " << FLAGS_dataset << std::endl;
  percival::Dataset dataset(FLAGS_dataset, dgraph.failure_mode_names(),
                            dgraph.test_names());
  std::cout << "Loaded " << dataset.size() << " samples" << std::endl;

  // const auto algorithm = Inference::ToAlgorithm(FLAGS_inference);
  // const auto fg = dgraph.factor_graph();
  // fg->inference_options.algorithm = algorithm;
  // fg->inference_options = Inference::DefaultInferenceOptions(algorithm);

  std::filesystem::path output_dir(FLAGS_output);
  const auto inference_output =
      output_dir / std::filesystem::path(kInferenceOutputFilename);
  const auto timing_output = output_dir / std::filesystem::path(kTimingOutputFilename);

  std::cout << "Saving results to: " << inference_output << std::endl;
  std::vector<std::string> output_headers = dgraph.failure_mode_names();
  std::ofstream ofs(inference_output);
  std::ofstream ofs_timing(timing_output);
  auto writer = csv::make_csv_writer(ofs);
  writer << output_headers;
  progressbar bar(dataset.size(), FLAGS_verbose);
  for (const auto sample : dataset) {
    auto t_start = std::chrono::high_resolution_clock::now();
    auto fi = dgraph.faultIdentification(sample.syndrome);
    auto t_end = std::chrono::high_resolution_clock::now();
    bar.update();
    std::vector<unsigned int> output_values;
    for (const auto &fm : output_headers) {
      output_values.push_back(fi[fm]);
    }
    writer << output_values;
    auto elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start);
    ofs_timing << elapsed.count() << std::endl;
  }

  return 0;
}