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
DEFINE_string(dataset, "", "Path to dataset [.csv]");
// DEFINE_string(dataset_output, "", "Path to dataset (output) [.csv]");
DEFINE_string(test_dataset, "", "Path to test dataset [.csv]");
// DEFINE_string(test_output, "", "Path to test dataset (output) [.csv]");
DEFINE_string(output, "", "Output Folder");
DEFINE_bool(randomize, false, "Randomize energies");

DEFINE_uint32(cd_gibbs_sweeps, 1000, "CD Number of Gibbs Sweeps");
DEFINE_uint32(cd_batch_size, 0, "CD Batch size");
DEFINE_uint32(cd_iterations, 30, "CD Iterations");
DEFINE_double(cd_stepsize, 0.1, "CD Stepsize");
DEFINE_uint32(ml_max_iterations, 100, "ML Max iterations");
DEFINE_double(ml_tolerance, 1.0e-8, "ML Tolerance");
DEFINE_double(ssvm_tolerance, 1.0e-8, "SSVM Tolerance");
DEFINE_double(ssvm_regularization, 0.01, "SSVM Regularization");
DEFINE_string(ssvm_opt_method, "bmrm", "SSVM Optimization Method");
DEFINE_uint32(ssvm_iterations, 100, "SSVM Iterations");

// clang-format off
static bool IsNonEmptyMessage(const char *flagname, const std::string &value) { return value[0] != '\0'; }
DEFINE_validator(model, &IsNonEmptyMessage);
DEFINE_validator(output, &IsNonEmptyMessage);
DEFINE_validator(dataset, &IsNonEmptyMessage);
// DEFINE_validator(gibbs_sweeps, [](const char *flagname, uint32_t value) { return value > 0; });
// DEFINE_validator(batch_size, [](const char *flagname, uint32_t value) { return value >= 0; });
// DEFINE_validator(stepsize, [](const char *flagname, double value) { return value > 0; });
// DEFINE_validator(iterations, [](const char *flagname, uint32_t value) { return value >= 0; });
// DEFINE_validator(max_iterations, [](const char *flagname, uint32_t value) { return value >= 0; });
// DEFINE_validator(tolerance, [](const char *flagname, double value) { return value > 0; });
// clang-format on

using FactorGraph = percival::factor_graph::grante::FactorGraph;
namespace Inference = percival::factor_graph::grante::inference;

// const std::string kModelOutputFilename = "trained_model.json";
const std::string kModelParamsFilename = "checkpoint.json";
const std::string kTrainInferenceOutputFilename = "train_inference.csv";
const std::string kTrainTimingOutputFilename = "train_timing.csv";
const std::string kTestInferenceOutputFilename = "test_inference.csv";
const std::string kTestTimingOutputFilename = "test_timing.csv";

int main(int argc, char *argv[]) {
  gflags::SetUsageMessage("Train a Diagnostic Factor Graph");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  gflags::ShutDownCommandLineFlags();

  std::cout << "Loading model: " << FLAGS_model << std::endl;
  percival::DiagnosticGraph<FactorGraph> dgraph;
  dgraph.fromJson(FLAGS_model);
  if (FLAGS_randomize) {
    dgraph.randomizeWeights();
  }
  dgraph.bake();
  if (FLAGS_verbose) {
    dgraph.summary();
    dgraph.print();
  }

  dgraph.factor_graph()->training_options.cd_gibbs_sweeps = FLAGS_cd_gibbs_sweeps;
  dgraph.factor_graph()->training_options.cd_batch_size = FLAGS_cd_batch_size;
  dgraph.factor_graph()->training_options.cd_stepsize = FLAGS_cd_stepsize;
  dgraph.factor_graph()->training_options.cd_iterations = FLAGS_cd_iterations;
  dgraph.factor_graph()->training_options.ml_max_iterations = FLAGS_ml_max_iterations;
  dgraph.factor_graph()->training_options.ml_tolerance = FLAGS_ml_tolerance;
  dgraph.factor_graph()->training_options.ssvm_tolerance = FLAGS_ssvm_tolerance;
  dgraph.factor_graph()->training_options.ssvm_regularization =
      FLAGS_ssvm_regularization;
  dgraph.factor_graph()->training_options.ssvm_opt_method = FLAGS_ssvm_opt_method;
  dgraph.factor_graph()->training_options.ssvm_iterations = FLAGS_ssvm_iterations;

  std::cout << "Loading dataset: " << FLAGS_dataset << std::endl;
  percival::Dataset dataset(FLAGS_dataset, dgraph.failure_mode_names(),
                            dgraph.test_names());
  std::cout << "Dataset size: " << dataset.size() << std::endl;
  std::cout << "Training model..." << std::endl;
  dgraph.train(dataset);
  std::cout << "training completed." << std::endl;

  std::filesystem::path output_dir(FLAGS_output);
  const auto train_dataset_output =
      output_dir / std::filesystem::path(kTrainInferenceOutputFilename);
  const auto train_dataset_timing_output =
      output_dir / std::filesystem::path(kTrainTimingOutputFilename);

  std::cout << "Running inference on Training dataset..." << std::endl;
  std::cout << "Saving inference results to: " << train_dataset_output << std::endl;
  std::cout << "Saving timing results to: " << train_dataset_timing_output << std::endl;
  std::vector<std::string> output_headers = dgraph.failure_mode_names();
  std::ofstream ofs_inference(train_dataset_output);
  std::ofstream ofs_timing(train_dataset_timing_output);
  auto fi_writer = csv::make_csv_writer(ofs_inference);
  fi_writer << output_headers;
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
    fi_writer << output_values;
    auto elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start);
    ofs_timing << elapsed.count() << std::endl;
  }
  std::cout << "Inference on Training dataset completed." << std::endl;

  if (!FLAGS_test_dataset.empty()) {
    const auto test_dataset_output =
        output_dir / std::filesystem::path(kTestInferenceOutputFilename);
    const auto test_dataset_timing_output =
        output_dir / std::filesystem::path(kTestTimingOutputFilename);
    std::cout << "Loading test dataset: " << FLAGS_test_dataset << std::endl;
    percival::Dataset test_dataset(FLAGS_test_dataset, dgraph.failure_mode_names(),
                                   dgraph.test_names());
    std::cout << "Loaded " << dataset.size() << " samples" << std::endl;
    std::cout << "Running inference on Testing dataset..." << std::endl;
    std::cout << "Saving inference results to: " << test_dataset_output << std::endl;
    std::cout << "Saving timing results to: " << test_dataset_timing_output
              << std::endl;
    std::vector<std::string> output_headers = dgraph.failure_mode_names();
    std::ofstream ofs_test_inference(test_dataset_output);
    std::ofstream ofs_test_timing(test_dataset_timing_output);
    auto fi_writer = csv::make_csv_writer(ofs_test_inference);
    fi_writer << output_headers;
    progressbar bar(test_dataset.size(), FLAGS_verbose);
    for (const auto sample : test_dataset) {
      auto t_start = std::chrono::high_resolution_clock::now();
      auto fi = dgraph.faultIdentification(sample.syndrome);
      auto t_end = std::chrono::high_resolution_clock::now();
      bar.update();
      std::vector<unsigned int> output_values;
      for (const auto &fm : output_headers) {
        output_values.push_back(fi[fm]);
      }
      fi_writer << output_values;
      auto elapsed =
          std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start);
      ofs_test_timing << elapsed.count() << std::endl;
    }
    std::cout << "Inference on Testing dataset completed." << std::endl;
  }

  // {
  //   const auto model_output = output_dir / std::filesystem::path(kModelOutputFilename);
  //   std::cout << "Saving model to: " << model_output << std::endl;
  //   dgraph.toJson(model_output);
  // }
  {
    const auto model_output = output_dir / std::filesystem::path(kModelParamsFilename);
    std::cout << "Saving params to: " << model_output << std::endl;
    dgraph.factor_graph()->save(model_output);
  }
  return 0;
}