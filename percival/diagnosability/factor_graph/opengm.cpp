#include "percival/diagnosability/factor_graph/opengm.h"

#include <iostream>

namespace percival {
namespace factor_graph {
namespace opengm {

namespace _internal {

unsigned int LinearIndexToVariableState(size_t ei, size_t var_index) {
  return ((ei / (1 << var_index)) % 2);
}

Model Copy(const ModifiedModel &mm) {
  FactorGraph::SpaceType space(mm.numberOfVariables(), 2);
  FactorGraph::Model gm(space);
  for (size_t f = 0; f < mm.numberOfFactors(); ++f) {
    auto phi = mm[f];
    if (phi.numberOfVariables() == 0) continue;
    std::vector<Index> scope;
    for (auto x = phi.variableIndicesBegin(); x != phi.variableIndicesEnd(); ++x)
      scope.push_back(*x);
    std::vector<double> weights;
    for (size_t ei = 0; ei < (1 << scope.size()); ++ei) {
      std::vector<Index> coord;
      for (size_t vi = 0; vi < scope.size(); ++vi)
        coord.push_back(LinearIndexToVariableState(ei, vi));
      weights.push_back(phi(coord.data()));
    }
    const std::vector<LabelType> shape(scope.size(), 2);
    ExplicitFunction fcn(shape.begin(), shape.end());
    for (size_t i = 0; i < weights.size(); ++i) fcn(i) = weights[i];
    auto fid = gm.addFunction(fcn);
    gm.addFactor(fid, scope.begin(), scope.end());
  }
  return gm;
}

}  // namespace _internal

FactorGraph::FactorGraph() {
  inference_options =
      inference::DefaultInferenceOptions(inference::Algorithm::kBeliefPropagation);
}

void FactorGraph::bake(DiagnosticGraph<FactorGraph> *dgraph) {
  dgraph_ = dgraph;
  if (dgraph_ == nullptr) {
    throw std::runtime_error("Diagnostic Factor Graph is not valid (nullptr).");
  }
  // the real baking is made on the fly (lazy baking)...
}

const FactorGraph::Model FactorGraph::lazy_bake_() const {
  if (dgraph_ == nullptr) {
    throw std::runtime_error("Diagnostic Factor Graph is not valid (nullptr).");
  }
  FactorGraph::SpaceType space(dgraph_->variables().size(), 2);
  FactorGraph::Model gm(space);
  // Add priors
  for (const auto &[varname, var] : dgraph_->variables()) {
    if (var.prior.has_value()) {
      const FactorGraph::LabelType shape[] = {2};
      FactorGraph::ExplicitFunction f(shape, shape + 1);
      f(0) = 1 - var.prior.value();
      f(1) = var.prior.value();
      auto fid = gm.addFunction(f);
      const Index var_idx[] = {var.index};
      gm.addFactor(fid, var_idx, var_idx + 1);
    }
  }
  // Add tests
  for (const auto &[testname, test] : dgraph_->tests()) {
    const std::size_t card = test.scope.size() + 1;
    const std::vector<unsigned int> shape(card, 2);
    FactorGraph::ExplicitFunction f(shape.begin(), shape.end());
    const auto w =
        test.weights.get(Weights::Convention::kDensity, Weights::Ordering::kReversed);
    for (std::size_t i = 0; i < w.size(); ++i) f(i) = w[i];
    std::vector<Index> scope;
    const auto vid = dgraph_->variables().at(testname).index;
    for (const auto &v : test.scope) scope.push_back(dgraph_->variables().at(v).index);
    scope.push_back(vid);
    auto fid = gm.addFunction(f);
    gm.addFactor(fid, scope.begin(), scope.end());
  }
  // Add relationships
  for (const auto &[relname, rel] : dgraph_->relationships()) {
    const std::vector<LabelType> shape(rel.scope.size(), 2);
    ExplicitFunction f(shape.begin(), shape.end());
    const auto w =
        rel.weights.get(Weights::Convention::kDensity, Weights::Ordering::kReversed);
    for (unsigned int i = 0; i < w.size(); ++i) f(i) = w[i];
    auto fid = gm.addFunction(f);
    std::vector<Index> scope;
    for (const auto &s : rel.scope) scope.push_back(dgraph_->variables().at(s).index);
    gm.addFactor(fid, scope.begin(), scope.end());
  }
  return gm;
}

const Model FactorGraph::lazy_bake_(const Syndrome &syn) const {
  if (dgraph_ == nullptr) {
    throw std::runtime_error("Diagnostic Factor Graph is not valid (nullptr).");
  }
  Model gm = lazy_bake_();
  ModelManipulator gmm(gm);
  for (const auto &[varname, s] : syn) {
    auto vid = dgraph_->variables().at(varname).index;
    gmm.fixVariable(vid, static_cast<LabelType>(s));
  }
  gmm.lock();
  gmm.buildModifiedModel();
  std::cout << "# Submodels: " << gmm.numberOfSubmodels() << std::endl;
  const ModifiedModel mdl = gmm.getModifiedModel();
  return _internal::Copy(mdl);
}

const Model FactorGraph::lazy_bake_(const SystemState &state) const {
  if (dgraph_ == nullptr) {
    throw std::runtime_error("Diagnostic Factor Graph is not valid (nullptr).");
  }
  Model gm = lazy_bake_();
  ModelManipulator gmm(gm);
  for (const auto &[varname, s] : state) {
    auto vid = dgraph_->variables().at(varname).index;
    gmm.fixVariable(vid, static_cast<LabelType>(s));
  }
  gmm.lock();
  gmm.buildModifiedModel();
  std::cout << "# Submodels: " << gmm.numberOfSubmodels() << std::endl;
  const ModifiedModel mdl = gmm.getModifiedModel();
  return _internal::Copy(mdl);
}

SystemState FactorGraph::faultIdentification(const Syndrome &syn) const {
  if (dgraph_ == nullptr)
    throw std::runtime_error("Diagnostic Factor Graph is not valid (nullptr).");
  auto gm = lazy_bake_(syn);
  auto est_state = inference::MAPInference(gm, inference_options);
  SystemState state;
  for (std::size_t var_id = 0; var_id < est_state.size(); ++var_id) {
    auto var = dgraph_->getVariable(var_id);
    if (var == std::nullopt)
      throw std::runtime_error("ERROR: Could not find variable " +
                               std::to_string(var_id));
    state[var.value()->name] = static_cast<FailureModeState>(est_state[var_id]);
  }
  return state;
}

// SystemState FactorGraph::marginalEstmation(const SystemState &state) const {
//   if (dgraph_ == nullptr) throw std::runtime_error("Diagnostic Factor Graph is not
//   valid (nullptr)."); auto gm = lazy_bake_(state); inference::Marginals(gm);
// }

void FactorGraph::print() const { std::cout << "Nothing to print." << std::endl; }

FailureProbability FactorGraph::getFailureProbability(const Variable &var) const {
  throw std::runtime_error("Not supported yet");
}
Weights FactorGraph::getWeights(const Test &test) const {
  throw std::runtime_error("Not supported yet");
}
Weights FactorGraph::getWeights(const Relationship &relationship) const {
  throw std::runtime_error("Not supported yet");
}

}  // namespace opengm
}  // namespace factor_graph
}  // namespace percival