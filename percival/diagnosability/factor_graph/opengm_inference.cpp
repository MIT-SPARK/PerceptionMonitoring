#include "percival/diagnosability/factor_graph/opengm_inference.h"

#include <iostream>
// #include <opengm/inference/gibbs.hxx>
#include <opengm/inference/icm.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/inference/trws/trws_trws.hxx>

namespace percival {
namespace factor_graph {
namespace opengm {
namespace inference {

Algorithm ToAlgorithm(const std::string& name) {
  if (name == "belief_propagation") {
    return Algorithm::kBeliefPropagation;
  } else if (name == "trwsi") {
    return Algorithm::kTRWSi;
  } else if (name == "icm") {
    return Algorithm::kICM;
  } else {
    throw std::invalid_argument("Unknown algorithm: " + name);
  }
}

std::string ToString(const Algorithm algorithm) {
  switch (algorithm) {
    case Algorithm::kBeliefPropagation:
      return "belief_propagation";
    case Algorithm::kTRWSi:
      return "trwsi";
    case Algorithm::kICM:
      return "icm";
  }
  return "unknown";
}

InferenceOptions DefaultInferenceOptions(const Algorithm algorithm) {
  InferenceOptions options;
  options.algorithm = algorithm;
  if (algorithm == Algorithm::kBeliefPropagation) {
    options.parameters.belief_propagation.max_terations = 100;
    options.parameters.belief_propagation.minimal_message_distance = 0.01;
    options.parameters.belief_propagation.damping = 0.8;
  } else if (algorithm == Algorithm::kICM) {
    options.parameters.trwsi.max_iterations = 100;
    options.parameters.trwsi.precision = 1e-12;
  }
  return options;
}

State MAPInference(const Model& model, const InferenceOptions& options) {
  typedef Model::IndependentFactorType IndependentFactor;

  State state;
  if (options.algorithm == Algorithm::kBeliefPropagation) {
    typedef ::opengm::BeliefPropagationUpdateRules<Model, OptimizationType> UpdateRules;
    typedef ::opengm::MessagePassing<Model, OptimizationType, UpdateRules, ::opengm::MaxDistance> LBP;
    LBP::Parameter parameter(options.parameters.belief_propagation.max_terations,
                             options.parameters.belief_propagation.minimal_message_distance,
                             options.parameters.belief_propagation.damping);
    LBP lbp(model, parameter);
    lbp.infer();
    lbp.arg(state);
  } else if (options.algorithm == Algorithm::kICM) {
    typedef ::opengm::ICM<Model, OptimizationType> ICM;
    ICM icm(model);
    icm.infer();
    icm.arg(state);
  } else if (options.algorithm == Algorithm::kTRWSi) {
    typedef ::opengm::TRWSi<Model, ::opengm::Maximizer> TRWSi;
    TRWSi::Parameter params(options.parameters.trwsi.max_iterations);
    params.precision_ = options.parameters.trwsi.precision;
    TRWSi trws(model, params);
    trws.infer();
    trws.arg(state);
  } else {
    throw std::invalid_argument("Unknown algorithm: " + ToString(options.algorithm));
  }
  return state;
}

// void Marginals(const Model& model) {
//   typedef ::opengm::Gibbs<Model, OptimizationType> Gibbs;
//   typedef ::opengm::GibbsMarginalVisitor<Gibbs> MarginalVisitor;
//   MarginalVisitor visitor(model);
//   for (auto i = 0; i < model.numberOfVariables(); ++i)
//     visitor.addMarginal(i);
//   Gibbs gibbs(model);
//   gibbs.infer(visitor);
//   for (auto v = 0; v < model.numberOfVariables(); ++v) {
//     std::cout << "P(x_" << v << "): ";
//     for (auto l = 0; l < model.numberOfLabels(v); ++l) {
//       const double p = static_cast<double>(visitor.marginal(v)(l)) / visitor.numberOfSamples();
//       std::cout << p << ' ';
//     }
//   }
// }

}  // namespace inference
}  // namespace opengm
}  // namespace factor_graph
}  // namespace percival