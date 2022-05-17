#pragma once

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/opengm.hxx>
#include <opengm/operations/multiplier.hxx>

#include "percival/diagnosability/dataset.h"
#include "percival/diagnosability/diagnostic_graph.h"
#include "percival/diagnosability/factor_graph/factor_graph.h"
#include "percival/diagnosability/factor_graph/opengm_inference.h"
#include "percival/diagnosability/typedefs.h"

namespace percival {
namespace factor_graph {
namespace opengm {

namespace _internal {

unsigned int LinearIndexToVariableState(size_t ei, size_t var_index);
Model Copy(const ModifiedModel &mm);

}  // namespace _internal

class FactorGraph : public FactorGraphBase {
 public:
  typedef double ValueType;                                                          // type used for values
  typedef unsigned int LabelType;                                                    // type used for labels
  typedef ::opengm::Multiplier OpType;                                               // operation used to combine terms
  typedef ::opengm::ExplicitFunction<ValueType, Index, LabelType> ExplicitFunction;  // shortcut for explicite function
  // typedef ::opengm::PottsFunction<ValueType, Index, LabelType> PottsFunction;        // shortcut for Potts function
  // typedef ::opengm::meta::TypeListGenerator<ExplicitFunction, PottsFunction>::type FunctionTypeList;
  typedef ::opengm::SimpleDiscreteSpace<> SpaceType;  // type used to define the feasible statespace
  typedef ::opengm::GraphicalModel<ValueType, OpType, ExplicitFunction, SpaceType> Model;  // type of the model
  typedef Model::FunctionIdentifier FunctionIdentifier;  // type of the function identifier

  inference::InferenceOptions inference_options;

 public:
  FactorGraph();
  bool isValid() const { return dgraph_ != nullptr; };
  void bake(DiagnosticGraph<FactorGraph> *dgraph);

  SystemState faultIdentification(const Syndrome &syn) const;
  // SystemState marginalEstmation(const SystemState &state) const;

  void train(const Dataset &dataset) { throw std::runtime_error("FactorGraph::train() not supported."); };
  void print() const;

  FailureProbability getFailureProbability(const Variable &var) const;
  Weights getWeights(const Test &test) const;
  Weights getWeights(const Relationship &relationship) const;

  const Capabilities getCapabilities() const {
    return Capabilities({Capabilities::INFERENCE, Capabilities::MAX_CARDINALITY});
  }

 private:
  const Model lazy_bake_() const;
  const Model lazy_bake_(const Syndrome &syn) const;
  const Model lazy_bake_(const SystemState &state) const;
  DiagnosticGraph<FactorGraph> *dgraph_;
};

}  // namespace opengm
}  // namespace factor_graph
}  // namespace percival