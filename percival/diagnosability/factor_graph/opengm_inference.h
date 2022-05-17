#pragma once

#include <string>
#include <vector>
#include "percival/diagnosability/factor_graph/opengm_typedefs.h"

namespace percival {
namespace factor_graph {
namespace opengm {
namespace inference {

enum class Algorithm {
  kBeliefPropagation,
  kICM,
  kTRWSi, // quite useless
};

struct TRWSiParameters {
  unsigned int max_iterations;
  double precision;
};

struct BeliefPropagationParameters {
  unsigned int max_terations;
  double minimal_message_distance;
  double damping;
};

struct InferenceOptions {
  Algorithm algorithm;
  union Parameters {
    TRWSiParameters trwsi;
    BeliefPropagationParameters belief_propagation;
  } parameters;
  bool verbose;
  InferenceOptions(){};
};

typedef std::vector<std::size_t> State;

Algorithm ToAlgorithm(const std::string& name);
std::string ToString(const Algorithm algorithm);
InferenceOptions DefaultInferenceOptions(const Algorithm algorithm);

State MAPInference(const Model& model, const InferenceOptions& options);
// void Marginals(const Model& model);

}  // namespace inference
}  // namespace opengm
}  // namespace factor_graph
}  // namespace percival