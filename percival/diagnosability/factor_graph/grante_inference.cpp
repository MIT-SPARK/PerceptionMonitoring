#include "percival/diagnosability/factor_graph/grante_inference.h"

#include "grante/AISInference.h"
#include "grante/BeliefPropagation.h"
#include "grante/BruteForceExactInference.h"
#include "grante/DiffusionInference.h"
#include "grante/FactorGraphStructurizer.h"
#include "grante/GibbsInference.h"
#include "grante/LinearProgrammingMAPInference.h"
#include "grante/MultichainGibbsInference.h"
#include "grante/NaiveMeanFieldInference.h"
#include "grante/SAMCInference.h"
#include "grante/SimulatedAnnealingInference.h"
#include "grante/StructuredMeanFieldInference.h"
#include "grante/TreeInference.h"

namespace percival {
namespace factor_graph {
namespace grante {
namespace inference {

InferenceOptions DefaultInferenceOptions(const Algorithm algorithm) {
  InferenceOptions options;
  options.algorithm = algorithm;
  options.verbose = false;
  if (algorithm == Algorithm::kDiffusion) {
    options.parameters.standard.max_iterations = 300;
    options.parameters.standard.tolerance = 1e-6;
  } else if (algorithm == Algorithm::kNaiveMeanField) {
    options.parameters.standard.max_iterations = 50;
    options.parameters.standard.tolerance = 1e-6;
  } else if (algorithm == Algorithm::kBeliefPropagation) {
    options.parameters.standard.max_iterations = 300;
    options.parameters.standard.tolerance = 1e-6;
  } else if (algorithm == Algorithm::kLinearProgramming) {
    options.parameters.standard.max_iterations = 100;
    options.parameters.standard.tolerance = 1e-6;
  } else if (algorithm == Algorithm::kSimulatedAnnealing) {
    options.parameters.simulated_annealing.step = 200;
    options.parameters.simulated_annealing.initial_temperature = 10;
    options.parameters.simulated_annealing.final_temperature = 0.01;
  }
  return options;
}

TrainingOptions DefaultTrainingOptions() {
  TrainingOptions options;
  options.cd_gibbs_sweeps = 1000;
  options.cd_batch_size = 0;
  options.cd_stepsize = 0.01;
  options.cd_iterations = 30;
  options.ml_max_iterations = 100;
  options.ml_tolerance = 1.0e-8;
  options.ssvm_tolerance = 1.0e-8;
  options.ssvm_regularization = 0.01;
  options.ssvm_opt_method = "bmrm";
  options.ssvm_iterations = 100;
  return options;
}

Algorithm ToAlgorithm(const std::string& name) {
  if (name == "tree") {
    return Algorithm::kTree;
  } else if (name == "diffusion") {
    return Algorithm::kDiffusion;
  } else if (name == "brute_force") {
    return Algorithm::kBruteForce;
  } else if (name == "naive_mean_field") {
    return Algorithm::kNaiveMeanField;
  } else if (name == "belief_propagation") {
    return Algorithm::kBeliefPropagation;
  } else if (name == "linear_programming") {
    return Algorithm::kLinearProgramming;
  } else if (name == "simulated_annealing") {
    return Algorithm::kSimulatedAnnealing;
  } else if (name == "gibbs") {
    return Algorithm::kGibbs;
  } else if (name == "ais") {
    return Algorithm::kAIS;
  } else if (name == "samc") {
    return Algorithm::kSAMC;
  } else {
    throw std::runtime_error("Unknown algorithm: " + name);
  }
}

std::string ToString(const Algorithm algorithm) {
  switch (algorithm) {
    case Algorithm::kTree:
      return "tree";
    case Algorithm::kDiffusion:
      return "diffusion";
    case Algorithm::kBruteForce:
      return "brute_force";
    case Algorithm::kNaiveMeanField:
      return "naive_mean_field";
    case Algorithm::kBeliefPropagation:
      return "belief_propagation";
    case Algorithm::kLinearProgramming:
      return "linear_programming";
    case Algorithm::kSimulatedAnnealing:
      return "simulated_annealing";
    case Algorithm::kGibbs:
      return "gibbs";
    case Algorithm::kAIS:
      return "ais";
    case Algorithm::kSAMC:
      return "samc";
  }
  return "unknown";
}

bool SupportsMAP(const Algorithm algorithm) {
  switch (algorithm) {
    case Algorithm::kTree:
    case Algorithm::kDiffusion:
    case Algorithm::kBruteForce:
    case Algorithm::kNaiveMeanField:
    case Algorithm::kBeliefPropagation:
    case Algorithm::kLinearProgramming:
    case Algorithm::kSimulatedAnnealing:
      return true;
    default:
      return false;
  }
}

std::shared_ptr<Grante::InferenceMethod> InferenceAlgorithm(
    const Grante::FactorGraph* fg, const InferenceOptions& options) {
  std::shared_ptr<Grante::InferenceMethod> inference;
  if (options.algorithm == Algorithm::kTree) {
    if (!Grante::FactorGraphStructurizer::IsForestStructured(fg))
      throw std::runtime_error(
          "TreeInference requires a forest-structured factor graph.");
    auto tf = std::make_unique<Grante::TreeInference>(fg);
    inference = std::move(tf);
  } else if (options.algorithm == Algorithm::kDiffusion) {
    auto di = std::make_unique<Grante::DiffusionInference>(fg);
    di->SetParameters(options.verbose, options.parameters.standard.max_iterations,
                      options.parameters.standard.tolerance);
    inference = std::move(di);
  } else if (options.algorithm == Algorithm::kBruteForce) {
    auto bfi = std::make_unique<Grante::BruteForceExactInference>(fg);
    inference = std::move(bfi);
  } else if (options.algorithm == Algorithm::kNaiveMeanField) {
    auto nmfi = std::make_unique<Grante::NaiveMeanFieldInference>(fg);
    nmfi->SetParameters(options.verbose, options.parameters.standard.tolerance,
                        options.parameters.standard.max_iterations);
    inference = std::move(nmfi);
  } else if (options.algorithm == Algorithm::kBeliefPropagation) {
    auto bp = std::make_unique<Grante::BeliefPropagation>(fg);
    bp->SetParameters(options.verbose, options.parameters.standard.max_iterations,
                      options.parameters.standard.tolerance);
    inference = std::move(bp);
  } else if (options.algorithm == Algorithm::kLinearProgramming) {
    auto lpi =
        std::make_unique<Grante::LinearProgrammingMAPInference>(fg, options.verbose);
    lpi->SetParameters(options.parameters.standard.max_iterations,
                       options.parameters.standard.tolerance);
    inference = std::move(lpi);
  } else if (options.algorithm == Algorithm::kSimulatedAnnealing) {
    auto sai =
        std::make_unique<Grante::SimulatedAnnealingInference>(fg, options.verbose);
    sai->SetParameters(options.parameters.simulated_annealing.step,
                       options.parameters.simulated_annealing.initial_temperature,
                       options.parameters.simulated_annealing.final_temperature);
    inference = std::move(sai);
  } else {
    throw std::runtime_error("Unknown algorithm: " + ToString(options.algorithm));
  }
  return inference;
}

State MAPInference(const Grante::FactorGraph* fg, const InferenceOptions& options) {
  auto inference = InferenceAlgorithm(fg, options);
  State state;
  inference->MinimizeEnergy(state);
  return state;
}

}  // namespace inference
}  // namespace grante
}  // namespace factor_graph
}  // namespace percival
