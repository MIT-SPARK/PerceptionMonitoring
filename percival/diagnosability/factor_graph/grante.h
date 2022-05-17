#pragma once

#include <memory>

#include "boost/bimap.hpp"
#include "grante/FactorGraph.h"
#include "percival/diagnosability/diagnostic_graph.h"
#include "percival/diagnosability/factor_graph/factor_graph.h"
#include "percival/diagnosability/factor_graph/grante_inference.h"
#include "percival/diagnosability/typedefs.h"

namespace percival {
namespace factor_graph {
namespace grante {

class FactorGraph : public FactorGraphBase {
 public:
  typedef boost::bimap<VarName, Index> FailureModeMap;

  FactorGraph();
  bool isValid() const { return model_ != nullptr; };
  void bake(DiagnosticGraph<FactorGraph> *dgraph);

  SystemState faultIdentification(const Syndrome &syn) const;
  void train(const Dataset &dataset);
  void trainCD(const Dataset &dataset);
  void trainML(const Dataset &dataset);
  void trainPW(const Dataset &dataset);
  void trainSSVM(const Dataset &dataset);
  // double eval(const SystemState &state, const Syndrome &syndrome) const;
  void print() const;

  FailureProbability getFailureProbability(const Variable &var) const;
  Weights getWeights(const Test &test) const;
  Weights getWeights(const Relationship &relationship) const;
  int numberOfParameters() const;

  void save(const std::string &filename) const;
  void load(DiagnosticGraph<FactorGraph> *dgraph, const std::string &filename);

  const Capabilities getCapabilities() const {
    return Capabilities({Capabilities::INFERENCE, Capabilities::TRAINING});
  }

 public:
  inference::InferenceOptions inference_options;
  inference::TrainingOptions training_options;

 private:
  DiagnosticGraph<FactorGraph> *dgraph_;
  std::unique_ptr<Grante::FactorGraphModel> model_;
  // the factor graphs only contains variables for the failure modes, so we need to
  // remap failure modes to new indices
  FailureModeMap failure_mode_map_;

  const std::string priorFactorName_(const Variable &var) const {
    return "pi_" + var.name;//std::to_string(var.index);
  };
  const std::string testFactorName_(const Test &test) const {
    return test.varname;
  };
  const std::string relationshipFactorName_(const Relationship &rel) const {
    return rel.varname;
  };
  std::unique_ptr<Grante::FactorGraph> bakeFactorGraph_(const Syndrome &syn) const;
};

}  // namespace grante
}  // namespace factor_graph
}  // namespace percival
