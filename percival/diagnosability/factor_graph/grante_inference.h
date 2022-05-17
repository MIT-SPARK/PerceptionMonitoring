#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "grante/FactorGraph.h"
#include "grante/InferenceMethod.h"

namespace percival {
namespace factor_graph {
namespace grante {
namespace inference {

enum class Algorithm {
  kTree,
  kDiffusion,
  kBruteForce,
  kNaiveMeanField,
  kBeliefPropagation,
  kLinearProgramming,
  kSimulatedAnnealing,
  kGibbs,
  kAIS,
  kSAMC,
};

struct StandardParameters {
  unsigned int max_iterations;
  double tolerance;
};

struct SimulatedAnnealingParameters {
  unsigned int step;
  double initial_temperature;
  double final_temperature;
};

struct InferenceOptions {
  Algorithm algorithm;
  union Parameters {
    StandardParameters standard;
    SimulatedAnnealingParameters simulated_annealing;
  } parameters;
  bool verbose;
  InferenceOptions(){};
};

struct TrainingOptions {
  // InferenceOptions inference_options;
  // gibbs_sweeps > 0: number of Gibbs sweeps to estimate model distributions sample
  unsigned int cd_gibbs_sweeps;
  // batch_size: number of instances to use for expectation.
  // If zero, all instances are used.  Typical values are 10, 100, 0.
  unsigned int cd_batch_size;
  // stepsize > 0: constant stepsize
  double cd_stepsize;
  unsigned int cd_iterations;
  unsigned int ml_max_iterations;
  double ml_tolerance;
  // Structured SVM
  double ssvm_regularization;
  std::string ssvm_opt_method;
  double ssvm_tolerance;
  unsigned int ssvm_iterations;
  TrainingOptions(){};
};

typedef std::vector<unsigned int> State;

Algorithm ToAlgorithm(const std::string& name);
std::string ToString(const Algorithm algorithm);

InferenceOptions DefaultInferenceOptions(const Algorithm algorithm);
TrainingOptions DefaultTrainingOptions();
bool SupportsMAP(const Algorithm algorithm);

std::shared_ptr<Grante::InferenceMethod> InferenceAlgorithm(
    const Grante::FactorGraph* fg, const InferenceOptions& options);
State MAPInference(const Grante::FactorGraph* fg, const InferenceOptions& options);

}  // namespace inference
}  // namespace grante
}  // namespace factor_graph
}  // namespace percival
