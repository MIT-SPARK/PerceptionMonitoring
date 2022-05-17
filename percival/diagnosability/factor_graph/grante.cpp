#include "percival/diagnosability/factor_graph/grante.h"

#include <fstream>

#include "grante/Conditioning.h"
#include "grante/ContrastiveDivergenceTraining.h"
#include "grante/Factor.h"
#include "grante/FactorConditioningTable.h"
#include "grante/FactorGraphModel.h"
#include "grante/FactorGraphPartialObservation.h"
#include "grante/FactorType.h"
#include "grante/MaximumPseudolikelihood.h"
#include "grante/NaivePiecewiseTraining.h"
#include "grante/NormalPrior.h"
#include "grante/StructuredSVM.h"
#include "nlohmann/json.hpp"
#include "percival/diagnosability/diagnostic_graph.h"
#include "percival/diagnosability/utils.h"
#include "percival/diagnosability/weights.h"

namespace percival {
namespace factor_graph {
namespace grante {

FactorGraph::FactorGraph()
    : inference_options(
          inference::DefaultInferenceOptions(inference::Algorithm::kBeliefPropagation)),
      training_options(inference::DefaultTrainingOptions()),
      dgraph_(nullptr),
      model_(nullptr) {}

void FactorGraph::print() const {
  if (model_ == nullptr) {
    std::cout << "Model is not baked yet.\n";
    return;
  }
  // std::cout << "Failure Modes:\n";
  // for (auto iter = failure_mode_map_.begin(), iend = failure_mode_map_.end();
  //      iter != iend; ++iter) {
  //   std::cout << iter->left << " <--> " << iter->right << std::endl;
  // }
}

void FactorGraph::bake(DiagnosticGraph<FactorGraph> *dgraph) {
  dgraph_ = dgraph;
  if (dgraph_ == nullptr) {
    throw std::runtime_error("Diagnostic Factor Graph is not valid (nullptr).");
  }
  // We bake just the model, the factor graph will be created on the fly for the
  // specific syndrome
  model_ = std::make_unique<Grante::FactorGraphModel>();
  // Add failure modes priors
  Index var_index = 0;
  for (const auto &[varname, var] : dgraph_->variables()) {
    if (var.type == Variable::Type::kModuleFailureMode ||
        var.type == Variable::Type::kOutputFailureMode) {
      failure_mode_map_.insert(FailureModeMap::value_type(varname, var_index++));
      if (var.prior.has_value()) {
        std::vector<double> w = {-std::log(var.prior.value()),
                                 -std::log(1. - var.prior.value())};
        const std::string factor_name = priorFactorName_(var);
        // Note: Grante::FactorGraph takes ownership of the FactorType
        model_->AddFactorType(new Grante::FactorType(factor_name, {2}, w));
      }
    }
  }
  // Initialize FactorType for Tests
  for (const auto &[testname, test] : dgraph_->tests()) {
    std::string factor_name = testFactorName_(test);
    Grante::FactorType *ft = model_->FindFactorType(factor_name);
    if (ft == nullptr) {
      const unsigned int card = test.scope.size();
      const std::vector<unsigned int> var_cards(card, 2);
      std::vector<double> w;
      test.weights.get(w, Weights::Convention::kEnergy, Weights::Ordering::kReversed);
      model_->AddFactorType(new Grante::FactorType(factor_name, var_cards, w));
    }
  }
  // Initialize FactorType for Relationships
  for (const auto &[relname, rel] : dgraph_->relationships()) {
    std::string factor_name = relationshipFactorName_(rel);
    Grante::FactorType *ft = model_->FindFactorType(factor_name);
    if (ft == nullptr) {
      const unsigned int card = rel.scope.size();
      const std::vector<unsigned int> var_cards(card, 2);
      std::vector<double> w;
      rel.weights.get(w, Weights::Convention::kEnergy, Weights::Ordering::kReversed);
      model_->AddFactorType(new Grante::FactorType(factor_name, var_cards, w));
    }
  }
}

std::unique_ptr<Grante::FactorGraph> FactorGraph::bakeFactorGraph_(
    const Syndrome &syn) const {
  if (dgraph_ == nullptr) {
    throw std::runtime_error("Diagnostic Factor Graph is not valid (nullptr).");
  }
  std::vector<double> data;
  std::vector<unsigned int> var_cards(dgraph_->num_failure_modes(), 2);
  // FactorGraph does not assume pointer ownership
  std::unique_ptr<Grante::FactorGraph> fg =
      std::make_unique<Grante::FactorGraph>(model_.get(), var_cards);
  std::vector<double> empty_data;
  // Add prior factors
  for (const auto &[varname, var] : dgraph_->variables()) {
    if (var.type == Variable::Type::kModuleFailureMode) {
      const std::string factor_name = priorFactorName_(var);
      auto *ft = model_->FindFactorType(factor_name);
      // Note: Grante::FactorGraph takes ownership of the factor
      const auto var_idx = failure_mode_map_.left.at(varname);
      fg->AddFactor(new Grante::Factor(ft, {var_idx}, empty_data));
    }
  }
  // Add test factors
  for (const auto &[testname, test] : dgraph_->tests()) {
    std::vector<Index> scope;
    for (const auto &v : test.scope) scope.push_back(failure_mode_map_.left.at(v));
    Grante::FactorType *ft = model_->FindFactorType(testFactorName_(test));
    std::vector<double> data(2, 0.0);
    data[static_cast<unsigned int>(syn.at(testname))] = 1.0;
    fg->AddFactor(new Grante::Factor(ft, scope, data));
  }
  // Add relationship factors
  for (const auto &[relname, rel] : dgraph_->relationships()) {
    std::vector<Index> scope;
    for (const auto &s : rel.scope) scope.push_back(failure_mode_map_.left.at(s));
    Grante::FactorType *ft = model_->FindFactorType(relationshipFactorName_(rel));
    fg->AddFactor(new Grante::Factor(ft, scope, empty_data));
  }
  fg->ForwardMap();
  return fg;
}

SystemState FactorGraph::faultIdentification(const Syndrome &syn) const {
  // Conditioning on observations
  // TestOutcomeMap test_outcomes;
  // std::vector<unsigned int> cond_var_set;
  // std::vector<unsigned int> cond_var_state;
  // for (const auto &obs : syn) {
  //   const auto v = dgraph_->variables().find(obs.first);
  //   if (v == dgraph_->variables().end()) throw std::runtime_error("ERROR: Observed
  //   variable not found in the model"); test_outcomes[v->second.index] = obs.second;
  // }
  // for (const auto &o : test_outcomes) {
  //   cond_var_set.push_back(o.first);
  //   cond_var_state.push_back(static_cast<unsigned int>(o.second));
  // }
  // Grante::FactorGraphPartialObservation partial_obs(cond_var_set, cond_var_state);
  // // var_new_to_orig: an index map such that
  // //    var_new_to_orig[new_factor_index] = original_factor_index
  // std::vector<unsigned int> var_new_to_orig;
  // Grante::FactorConditioningTable conditioning_table;
  // Grante::FactorGraph *fg_cond =
  //     Grante::Conditioning::ConditionFactorGraph(&conditioning_table, fg_.get(),
  //     &partial_obs, var_new_to_orig);
  // fg_cond->ForwardMap();
  auto fg_cond = bakeFactorGraph_(syn);

  // Inference
  SystemState state;
  auto est_state = inference::MAPInference(fg_cond.get(), inference_options);
  for (std::size_t i = 0; i < est_state.size(); i++) {
    // auto v = dgraph_->getVariable(var_new_to_orig[i]);
    // if (v == std::nullopt) throw std::runtime_error("ERROR: Could not find variable "
    // + std::to_string(i)); state[v.value()->name] =
    // static_cast<FailureModeState>(est_state[i]);
    const auto var_idx = static_cast<Index>(i);
    const auto var_val = static_cast<FailureModeState>(est_state[i]);
    state[failure_mode_map_.right.at(var_idx)] = var_val;
  }
  // delete fg_cond;
  return state;
}

void FactorGraph::train(const Dataset &train_data) {
  // if (training_options.ml_max_iterations > 0) trainPW(train_data);
  if (training_options.ml_max_iterations > 0) trainML(train_data);
  if (training_options.cd_iterations > 0) trainCD(train_data);
  if (training_options.ssvm_iterations > 0) trainSSVM(train_data);
  std::cout << "Grante Training complete" << std::endl;
}

void FactorGraph::trainML(const Dataset &train_data) {
  std::vector<Grante::ParameterEstimationMethod::labeled_instance_type> training_data;
  std::vector<Grante::InferenceMethod *> inference_methods;
  // std::map<Index, VarName, std::less<Index>> variables;
  std::vector<Grante::FactorGraphObservation *> observations;
  std::vector<std::unique_ptr<Grante::FactorGraph>> factorgraphs;
  std::vector<std::shared_ptr<Grante::InferenceMethod>> inference_methods_ptr;

  for (const auto &sample : train_data) {
    std::vector<unsigned int> data;
    // failure_mode_map_.right behaves as std::map<Index, VarName>, ordering by Index
    for (const auto &x : failure_mode_map_.right) {
      auto val = sample.at(x.second);
      if (val.has_value())
        data.push_back(val.value());
      else
        throw std::runtime_error("Missing value for variable " + x.first);
    }
    auto obs = new Grante::FactorGraphObservation(data);
    auto fg = bakeFactorGraph_(sample.syndrome);
    inference_methods_ptr.push_back(
        inference::InferenceAlgorithm(fg.get(), inference_options));
    training_data.push_back(
        Grante::ParameterEstimationMethod::labeled_instance_type(fg.get(), obs));
    // Save obs pointer to vector so we can delete it later
    observations.push_back(obs);
    // move ownership to vector, ensure fg does not go out of scope at end of loop
    factorgraphs.push_back(std::move(fg));
    inference_methods.push_back(inference_methods_ptr.back().get());
  }

  Grante::MaximumPseudolikelihood ml(model_.get());
  ml.SetupTrainingData(training_data, inference_methods);
  ml.Train(training_options.ml_tolerance, training_options.ml_max_iterations);

  for (const auto &x : observations) delete x;
  return;
}

void FactorGraph::trainCD(const Dataset &train_data) {
  std::vector<Grante::ParameterEstimationMethod::labeled_instance_type> training_data;
  std::vector<Grante::InferenceMethod *> inference_methods;
  // std::map<Index, VarName, std::less<Index>> variables;
  std::vector<Grante::FactorGraphObservation *> observations;
  std::vector<std::unique_ptr<Grante::FactorGraph>> factorgraphs;
  std::vector<std::shared_ptr<Grante::InferenceMethod>> inference_methods_ptr;

  for (const auto &sample : train_data) {
    std::vector<unsigned int> data;
    // failure_mode_map_.right behaves as std::map<Index, VarName>, ordering by Index
    for (const auto &x : failure_mode_map_.right) {
      auto val = sample.at(x.second);
      if (val.has_value())
        data.push_back(val.value());
      else
        throw std::runtime_error("Missing value for variable " + x.first);
    }
    auto obs = new Grante::FactorGraphObservation(data);
    auto fg = bakeFactorGraph_(sample.syndrome);
    // inference_methods_ptr.push_back(
    //     inference::InferenceAlgorithm(fg.get(), training_options.inference_options));
    training_data.push_back(
        Grante::ParameterEstimationMethod::labeled_instance_type(fg.get(), obs));
    // Save obs pointer to vector so we can delete it later
    observations.push_back(obs);
    // move ownership to vector, ensure fg does not go out of scope at end of loop
    factorgraphs.push_back(std::move(fg));
    // inference_methods.push_back(inference_methods_ptr.back().get());
  }

  Grante::ContrastiveDivergenceTraining cdt(
      model_.get(), training_options.cd_gibbs_sweeps, training_options.cd_batch_size,
      training_options.cd_stepsize);
  cdt.SetupTrainingData(training_data, inference_methods);
  cdt.Train(0, training_options.cd_iterations);
  // Grante::MaximumPseudolikelihood mple(model_.get());
  // mple.SetupTrainingData(training_data, inference_methods);
  // mple.Train(training_options.tolerance, training_options.max_iterations);

  for (const auto &x : observations) delete x;
  return;
}

void FactorGraph::trainPW(const Dataset &train_data) {
  std::vector<Grante::ParameterEstimationMethod::labeled_instance_type> training_data;
  std::vector<Grante::InferenceMethod *> inference_methods;
  // std::map<Index, VarName, std::less<Index>> variables;
  std::vector<Grante::FactorGraphObservation *> observations;
  std::vector<std::unique_ptr<Grante::FactorGraph>> factorgraphs;
  std::vector<std::shared_ptr<Grante::InferenceMethod>> inference_methods_ptr;

  for (const auto &sample : train_data) {
    std::vector<unsigned int> data;
    // failure_mode_map_.right behaves as std::map<Index, VarName>, ordering by Index
    for (const auto &x : failure_mode_map_.right) {
      auto val = sample.at(x.second);
      if (val.has_value())
        data.push_back(val.value());
      else
        throw std::runtime_error("Missing value for variable " + x.first);
    }
    auto obs = new Grante::FactorGraphObservation(data);
    auto fg = bakeFactorGraph_(sample.syndrome);
    inference_methods_ptr.push_back(
        inference::InferenceAlgorithm(fg.get(), inference_options));
    training_data.push_back(
        Grante::ParameterEstimationMethod::labeled_instance_type(fg.get(), obs));
    // Save obs pointer to vector so we can delete it later
    observations.push_back(obs);
    // move ownership to vector, ensure fg does not go out of scope at end of loop
    factorgraphs.push_back(std::move(fg));
    inference_methods.push_back(inference_methods_ptr.back().get());
  }

  Grante::NaivePiecewiseTraining pw(model_.get());
  pw.SetupTrainingData(training_data, inference_methods);
  pw.Train(training_options.ml_tolerance, training_options.ml_max_iterations);

  for (const auto &x : observations) delete x;
  return;
}

void FactorGraph::trainSSVM(const Dataset &train_data) {
  std::vector<Grante::ParameterEstimationMethod::labeled_instance_type> training_data;
  std::vector<Grante::InferenceMethod *> inference_methods;
  // std::map<Index, VarName, std::less<Index>> variables;
  std::vector<Grante::FactorGraphObservation *> observations;
  std::vector<std::unique_ptr<Grante::FactorGraph>> factorgraphs;
  std::vector<std::shared_ptr<Grante::InferenceMethod>> inference_methods_ptr;

  for (const auto &sample : train_data) {
    std::vector<unsigned int> data;
    // failure_mode_map_.right behaves as std::map<Index, VarName>, ordering by Index
    for (const auto &x : failure_mode_map_.right) {
      auto val = sample.at(x.second);
      if (val.has_value())
        data.push_back(val.value());
      else
        throw std::runtime_error("Missing value for variable " + x.first);
    }
    auto obs = new Grante::FactorGraphObservation(data);
    auto fg = bakeFactorGraph_(sample.syndrome);
    inference_methods_ptr.push_back(
        inference::InferenceAlgorithm(fg.get(), inference_options));
    training_data.push_back(
        Grante::ParameterEstimationMethod::labeled_instance_type(fg.get(), obs));
    // Save obs pointer to vector so we can delete it later
    observations.push_back(obs);
    // move ownership to vector, ensure fg does not go out of scope at end of loop
    factorgraphs.push_back(std::move(fg));
    inference_methods.push_back(inference_methods_ptr.back().get());
  }

  Grante::StructuredSVM trainer(model_.get(), training_options.ssvm_regularization,
                                training_options.ssvm_opt_method);
  trainer.SetupTrainingData(training_data, inference_methods);
  for (const auto ft : model_->FactorTypes()) {
    trainer.AddPrior(ft->Name(), new Grante::NormalPrior(10.0, ft->Weights().size()));
  }
  trainer.Train(training_options.ssvm_tolerance, training_options.ssvm_iterations);

  // for (const auto &x : observations) delete x;
  return;
}

FailureProbability FactorGraph::getFailureProbability(const Variable &var) const {
  auto pi_name = priorFactorName_(var);
  auto pi = model_->FindFactorType(pi_name);
  if (pi != nullptr)
    return FailureProbability(std::exp(-pi->Weights()[1]));
  else
    return FailureProbability({});
}

Weights FactorGraph::getWeights(const Test &test) const {
  auto phi_name = testFactorName_(test);
  auto phi = model_->FindFactorType(phi_name);
  if (phi != nullptr)
    return Weights(phi->Weights(), Weights::Convention::kEnergy,
                   Weights::Ordering::kReversed);
  else
    throw std::runtime_error("ERROR: Could not find test factor.");
}

Weights FactorGraph::getWeights(const Relationship &relationship) const {
  auto phi_name = relationshipFactorName_(relationship);
  auto phi = model_->FindFactorType(phi_name);
  if (phi != nullptr)
    return Weights(phi->Weights(), Weights::Convention::kEnergy,
                   Weights::Ordering::kReversed);
  else
    throw std::runtime_error("ERROR: Could not find test factor.");
}

// double FactorGraph::eval(const SystemState &state, const Syndrome &syndrome) const {
//   auto fg = bakeFactorGraph_(syndrome);
//   std::map<Index, unsigned int, std::less<Index>> assignment_map;
//   for (const auto &v : dgraph_->variables()) {
//     if (v.second.type == Variable::Type::kTest)
//       assignment_map[v.second.index] =
//           static_cast<unsigned int>(syndrome.find(v.first)->second);
//     else
//       assignment_map[v.second.index] =
//           static_cast<unsigned int>(state.find(v.first)->second);
//   }
//   std::vector<unsigned int> assignment;
//   for (const auto &[index, v] : assignment_map) {
//     assignment.push_back(v);
//   }
//   return fg->EvaluateEnergy(assignment);
// }

int FactorGraph::numberOfParameters() const {
  int num_params = 0;
  for (const auto ft : model_->FactorTypes()) {
    num_params += ft->Weights().size();
  }
  return num_params;
}

void FactorGraph::save(const std::string &filename) const {
  nlohmann::json checkpoint;
  // Save factor types
  checkpoint["FactorTypes"] = nlohmann::json::array();
  for (const auto factor_type : model_->FactorTypes()) {
    nlohmann::json ft;
    ft["Name"] = factor_type->Name();
    ft["Weights"] = nlohmann::json::array();
    for (const auto w : factor_type->Weights()) {
      ft["Weights"].push_back(w);
    }
    checkpoint["FactorTypes"].push_back(ft);
  }
  // Save Failure Mode Map
  // failure_mode_map_.right behaves as std::map<Index, VarName>, ordering by Index
  checkpoint["FailureIndex"] = nlohmann::json::array();
  for (const auto &x : failure_mode_map_.right) {
    nlohmann::json v;
    v["Index"] = x.first;
    v["Name"] = x.second;
    checkpoint["FailureIndex"].push_back(v);
  }
  // Save to file
  std::ofstream o(filename);
  o << std::setw(2) << checkpoint << std::endl;
}

void FactorGraph::load(DiagnosticGraph<FactorGraph> *dgraph,
                       const std::string &filename) {
  dgraph_ = dgraph;
  std::ifstream checkpoint_json(filename);
  nlohmann::json checkpoint;
  checkpoint_json >> checkpoint;
  for (auto &x : checkpoint["FailureIndex"]) {
    VarName varname = x["Name"].get<VarName>();
    Index var_index = x["Index"].get<Index>();
    failure_mode_map_.insert(FailureModeMap::value_type(varname, var_index));
  }
  // PreLoad factor types
  std::unordered_map<std::string, std::vector<double>> factor_types;
  for (auto &x : checkpoint["FactorTypes"]) {
    std::string name = x["Name"].get<std::string>();
    std::vector<double> weights;
    for (auto &w : x["Weights"]) weights.push_back(w.get<double>());
    factor_types.insert(std::make_pair(name, weights));
  }
  /////////////////////////////////////////////////////////////////////////////////////
  model_ = std::make_unique<Grante::FactorGraphModel>();
  // Priors
  for (const auto &[varname, var] : dgraph_->variables()) {
    if (var.type == Variable::Type::kModuleFailureMode ||
        var.type == Variable::Type::kOutputFailureMode) {
      const std::string factor_name = priorFactorName_(var);
      const auto it = factor_types.find(factor_name);
      if (it != factor_types.end()) {
        model_->AddFactorType(new Grante::FactorType(factor_name, {2}, it->second));
      }
    }
  }
  // Initialize FactorType for Tests
  for (const auto &[testname, test] : dgraph_->tests()) {
    std::string factor_name = testFactorName_(test);
    Grante::FactorType *ft = model_->FindFactorType(factor_name);
    if (ft == nullptr) {
      const unsigned int card = test.scope.size();
      const std::vector<unsigned int> var_cards(card, 2);
      const auto weights = factor_types.at(factor_name);
      model_->AddFactorType(new Grante::FactorType(factor_name, var_cards, weights));
    } else {
      throw std::runtime_error("ERROR: Factor type `" + factor_name +
                               "` already exists.");
    }
  }
  // Initialize FactorType for Relationships
  for (const auto &[relname, rel] : dgraph_->relationships()) {
    std::string factor_name = relationshipFactorName_(rel);
    Grante::FactorType *ft = model_->FindFactorType(factor_name);
    if (ft == nullptr) {
      const unsigned int card = rel.scope.size();
      const std::vector<unsigned int> var_cards(card, 2);
      const auto weights = factor_types.at(factor_name);
      model_->AddFactorType(new Grante::FactorType(factor_name, var_cards, weights));
    } else {
      throw std::runtime_error("ERROR: Factor type `" + factor_name +
                               "` already exists.");
    }
  }
}

}  // namespace grante
}  // namespace factor_graph
}  // namespace percival