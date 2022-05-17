#pragma once

#include <iostream>
#include <memory>

#include "percival/diagnosability/diagnostic_graph.h"
#include "percival/diagnosability/factor_graph/factor_graph.h"
#include "percival/diagnosability/utils.h"

namespace percival {
namespace factor_graph {
namespace dummy {
/**
 * @brief A dummy factor graph mainly used for testing.
 *
 */
class FactorGraph : public FactorGraphBase {
 public:
  FactorGraph() { dgraph_ = nullptr; };
  void bake(DiagnosticGraph<FactorGraph> *dgraph) { dgraph_ = dgraph; };

  SystemState faultIdentification(const Syndrome &syn) const {
    SystemState state;
    for (const auto &[testname, test] : dgraph_->tests()) {
      for (const auto &varname : test.scope)
        state[varname] =
            (syn.at(testname) == TestOutcome::FAIL) ? FailureModeState::ACTIVE : FailureModeState::INACTIVE;
    }
    return state;
  };
  void train(const Dataset &dataset) { return; };
  void print() const { std::cout << "Nothing to print." << std::endl; };

  FailureProbability getFailureProbability(const Variable &var) const {
    if (var.type == Variable::Type::kModuleFailureMode)
      return 0.5;
    else
      return {};
  };
  Weights getWeights(const Test &test) const { return utils::TestWeightsGenerator(test, &utils::IdealTest); };
  Weights getWeights(const Relationship &relationship) const {
    return utils::WeightsGenerator(relationship.scope, &utils::AllFailOrNone);
  };

  bool isValid() const { return dgraph_ != nullptr; };

  const Capabilities getCapabilities() const {
    return Capabilities({Capabilities::INFERENCE, Capabilities::TRAINING});
  }

 private:
  DiagnosticGraph<FactorGraph> *dgraph_;
};

}  // namespace dummy
}  // namespace factor_graph
}  // namespace percival
