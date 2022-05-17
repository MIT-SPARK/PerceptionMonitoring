#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "nlohmann/json.hpp"
#include "percival/diagnosability/diagnostic_graph.h"
#include "percival/diagnosability/factor_graph/factor_graph.h"
#include "percival/diagnosability/utils.h"

namespace percival {

template <typename FG>
void DiagnosticGraph<FG>::randomizeWeights() {
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (auto &[varname, var] : variables_) {
    if (var.prior.has_value()) var.prior = dist(mt);
  }
  for (auto &[testname, test] : tests_) test.weights.randomize();
  for (auto &[relname, rel] : relationships_) rel.weights.randomize();
}

template <typename FG>
DiagnosticGraph<FG>::DiagnosticGraph() : max_failure_cardinality_({}) {
  static_assert(std::is_base_of<factor_graph::FactorGraphBase, FG>::value,
                "FG must derive from FactorGraphBase");
  fg_ = std::make_unique<FG>();
};

template <typename FG>
template <typename T>
DiagnosticGraph<FG>::DiagnosticGraph(const DiagnosticGraph<T> &rhs) {
  for (const auto &var : rhs.variables()) {
    if (var.second.type != Variable::Type::kTest) {
      variables_[var.first] = var.second;
    }
  }
  for (const auto &test : rhs.tests()) addTest_(test.second);
  for (const auto &rel : rhs.relationships()) addRelationship_(rel.second);
  auto max_card = rhs.max_failure_cardinality();
  if (max_card.has_value())
    max_failure_cardinality(max_card.value());
  else
    max_failure_cardinality_ = {};
  fg_ = std::make_unique<FG>();
}

template <class FG>
void DiagnosticGraph<FG>::addFailureMode(const VarName &name,
                                         const FailureModeType &type,
                                         const FailureProbability prior) {
  // TODO: not sure if this is really required, in any case it should be allowed and
  // index to be re-assigned so that tests have always index higher than failure modes
  if (!tests_.empty())
    throw std::runtime_error("Cannot add failure modes after tests have been added");
  if (variables_.find(name) != variables_.end())
    throw std::runtime_error("Failure Mode already exists");
  const Variable::Type var_type = (type == FailureModeType::MODULE)
                                      ? Variable::Type::kModuleFailureMode
                                      : Variable::Type::kOutputFailureMode;
  variables_[name] = Variable(name, variables_.size(), var_type, prior);
  return;
}

template <class FG>
void DiagnosticGraph<FG>::addTest(const Name &name, const VarName &varname,
                                  const std::vector<VarName> &scope,
                                  const std::vector<double> &weights) {
  Test test(name, varname, tests_.size(), scope, Weights(weights));
  addTest_(test);
  return;
}

template <class FG>
void DiagnosticGraph<FG>::addTest(const Name &name, const VarName &varname,
                                  const std::vector<VarName> &scope,
                                  double (*gen)(const TestOutcome,
                                                const SystemState &)) {
  Test test(name, varname, tests_.size(), scope, Weights());
  test.weights = utils::TestWeightsGenerator(test, gen);
  addTest_(test);
  return;
}

template <class FG>
void DiagnosticGraph<FG>::addTest_(const Test &test) {
  if (tests_.find(test.varname) != tests_.end())
    throw std::runtime_error("Test already exists");
  if (variables_.find(test.varname) != variables_.end())
    throw std::runtime_error("There exist a variable with the same name");
  // Validate scope
  if (test.scope.size() == 0) throw std::runtime_error("Test scope must be non-empty");
  for (const auto &s : test.scope) {
    const auto var_it = variables_.find(s);
    if (var_it == variables_.end())
      throw std::runtime_error("Scope contains unknown failure mode");
    else if (var_it->second.type == Variable::Type::kTest)
      throw std::runtime_error("Scope contains a test");
  }
  // Validate weights
  const unsigned int table_size = 1 << (test.scope.size() + 1);
  if (test.weights.size() != table_size)
    throw std::runtime_error("Test weights must be of size 2^(scope.size()+1)");
  tests_[test.varname] = test;
  variables_[test.varname] =
      Variable(test.varname, variables_.size(), Variable::Type::kTest);
}

template <class FG>
void DiagnosticGraph<FG>::addRelationship(const VarName &name, const VarName &varname,
                                          const std::vector<VarName> &scope,
                                          const std::vector<double> &weights) {
  Relationship rel(name, varname, scope, weights);
  addRelationship_(rel);
  return;
}

template <class FG>
void DiagnosticGraph<FG>::addRelationship(const VarName &name, const VarName &varname,
                                          const std::vector<VarName> &scope,
                                          double (*gen)(const SystemState &)) {
  Relationship rel(name, varname, scope, {});
  rel.weights = utils::WeightsGenerator(scope, gen);
  addRelationship_(rel);
  return;
}

template <class FG>
void DiagnosticGraph<FG>::addRelationship_(const Relationship &relationship) {
  if (relationships_.find(relationship.varname) != relationships_.end())
    throw std::runtime_error("Relationship already exists");
  // Validate scope
  for (const auto &s : relationship.scope) {
    if (variables_.find(s) == variables_.end())
      throw std::runtime_error("Scope contains unknown failure modes");
  }
  // Validate weights
  const unsigned int table_size = 1 << (relationship.scope.size());
  if (relationship.weights.size() != table_size)
    throw std::runtime_error("Relationship weights must be of size 2^(scope.size())");
  relationships_[relationship.varname] = relationship;
  return;
}

template <class FG>
void DiagnosticGraph<FG>::max_failure_cardinality(unsigned int card) {
  if (card < 0)
    max_failure_cardinality_ = {};
  else
    max_failure_cardinality_ = card;
  return;
}

template <class FG>
const std::optional<unsigned int> DiagnosticGraph<FG>::max_failure_cardinality() const {
  return max_failure_cardinality_;
}

template <class FG>
void DiagnosticGraph<FG>::summary() const {
  int num_modules_failure_modes = std::count_if(
      variables_.begin(), variables_.end(), [](std::pair<VarName, Variable> p) {
        return (p.second.type == Variable::Type::kModuleFailureMode);
      });
  int num_output_failure_modes = std::count_if(
      variables_.begin(), variables_.end(), [](std::pair<VarName, Variable> p) {
        return (p.second.type == Variable::Type::kOutputFailureMode);
      });
  std::cout << "Summary of DiagnosticGraph" << std::endl;
  std::cout << "Number of variables: " << variables_.size() << std::endl;
  std::cout << "Number of module failure modes: " << num_modules_failure_modes
            << std::endl;
  std::cout << "Number of output failure modes: " << num_output_failure_modes
            << std::endl;
  std::cout << "Number of tests: " << tests_.size() << std::endl;
  if (max_failure_cardinality_.has_value()) {
    std::cout << "Max cardinality: " << max_failure_cardinality_.value() << std::endl;
  } else {
    std::cout << "Max cardinality: N/A" << std::endl;
  }
  std::cout << "Failure modes:" << std::endl;
  for (const auto &f : variables_) {
    if (f.second.type == Variable::Type::kModuleFailureMode or
        f.second.type == Variable::Type::kOutputFailureMode) {
      const auto type =
          (f.second.type == Variable::Type::kModuleFailureMode) ? "module" : "output";
      std::cout << "  (" << f.second.index << ") [" << type << "] " << f.first
                << std::endl;
    }
  }
  std::cout << "Tests:" << std::endl;
  for (const auto &t : tests_) {
    const auto vid = variables_.at(t.first).index;
    std::cout << "  (" << t.second.index << ") " << t.first << " (varid: " << vid
              << ") on  [ ";
    for (const auto &s : t.second.scope) {
      std::cout << s << ", ";
    }
    std::cout << "]" << std::endl;
  }
  return;
}

template <class FG>
void DiagnosticGraph<FG>::print() const {
  fg_->print();
  return;
}

template <class FG>
const std::vector<VarName> DiagnosticGraph<FG>::failure_mode_names() const {
  std::vector<VarName> names;
  for (const auto &f : variables_) {
    if (f.second.type == Variable::Type::kModuleFailureMode or
        f.second.type == Variable::Type::kOutputFailureMode) {
      names.push_back(f.first);
    }
  }
  return names;
}

template <class FG>
const std::size_t DiagnosticGraph<FG>::num_failure_modes() const {
  std::size_t num_failure_modes = 0;
  for (const auto &fm : variables_) {
    if (fm.second.type == Variable::Type::kModuleFailureMode ||
        fm.second.type == Variable::Type::kOutputFailureMode) {
      num_failure_modes++;
    }
  }
  return num_failure_modes;
}

template <class FG>
const std::vector<VarName> DiagnosticGraph<FG>::test_names() const {
  std::vector<VarName> names;
  for (const auto &t : tests_) names.push_back(t.first);
  return names;
}

template <class FG>
std::optional<const Variable *> DiagnosticGraph<FG>::getVariable(const Index id) const {
  for (const auto &v : variables_) {
    if (v.second.index == id) return &v.second;
  }
  return {};
}

template <class FG>
std::optional<const Variable *> DiagnosticGraph<FG>::getVariable(
    const VarName &name) const {
  auto it = variables_.find(name);
  if (it == variables_.end()) return {};
  return &it->second;
}

template <class FG>
void DiagnosticGraph<FG>::fromJson(const std::string &filepath) {
  std::ifstream mdl_json(filepath);
  nlohmann::json j;
  mdl_json >> j;

  for (auto &x : j["variables"]) {
    FailureProbability prior = {};
    VarName varname = x["varname"].get<VarName>();
    if (x["prior"] != nlohmann::detail::value_t::null) prior = x["prior"].get<double>();
    if (x["type"] == "module")
      this->addFailureMode(varname, FailureModeType::MODULE, prior);
    else if (x["type"] == "output")
      this->addFailureMode(varname, FailureModeType::OUTPUT, prior);
  }
  for (auto &x : j["tests"]) {
    Name test_name = x["name"].get<Name>();
    VarName varname = x["varname"].get<VarName>();
    std::vector<VarName> scope;
    for (auto &y : x["scope"]) scope.push_back(y.get<VarName>());
    std::vector<double> weights;
    for (auto &y : x["densities"]) weights.push_back(y.get<double>());
    this->addTest(test_name, varname, scope, weights);
  }
  for (auto &x : j["relationships"]) {
    Name name = x["name"].get<Name>();
    VarName varname = x["varname"].get<VarName>();
    std::vector<VarName> scope;
    for (auto &y : x["scope"]) scope.push_back(y.get<VarName>());
    std::vector<double> weights;
    for (auto &y : x["densities"]) weights.push_back(y.get<double>());
    this->addRelationship(name, varname, scope, weights);
  }
}

template <class FG>
void DiagnosticGraph<FG>::toJson(const std::string &filepath) const {
  nlohmann::json j;
  j["variables"] = nlohmann::json::array();
  for (const auto &v : variables_) {
    if (v.second.type == Variable::Type::kTest) continue;
    nlohmann::json var;
    var["varname"] = v.second.name;
    var["index"] = v.second.index;
    var["type"] =
        (v.second.type == Variable::Type::kModuleFailureMode) ? "module" : "output";
    if (v.second.prior.has_value())
      var["prior"] = v.second.prior.value();
    else
      var["prior"] = nullptr;
    j["variables"].push_back(var);
  }
  j["tests"] = nlohmann::json::array();
  for (const auto &[tname, t] : tests_) {
    nlohmann::json test;
    test["type"] = "test";
    test["name"] = t.name;
    test["varname"] = t.varname;
    test["index"] = t.index;
    test["scope"] = nlohmann::json::array();
    for (const auto &v : t.scope) test["scope"].push_back(v);
    test["densities"] = nlohmann::json::array();
    auto weights =
        t.weights.get(Weights::Convention::kDensity, Weights::Ordering::kNatural);
    for (const auto &w : weights) test["densities"].push_back(w);
    j["tests"].push_back(test);
  }
  j["relationships"] = nlohmann::json::array();
  for (const auto &[rnane, r] : relationships_) {
    nlohmann::json rel;
    rel["type"] = "constraint";
    rel["name"] = r.name;
    rel["varname"] = r.varname;
    rel["scope"] = nlohmann::json::array();
    for (const auto &v : r.scope) rel["scope"].push_back(v);
    rel["densities"] = nlohmann::json::array();
    auto weights = r.weights.get(Weights::Convention::kDensity,
                                        Weights::Ordering::kNatural);
    for (const auto &w : weights) rel["densities"].push_back(w);
    j["relationships"].push_back(rel);
  }
  std::ofstream o(filepath);
  o << std::setw(2) << j << std::endl;
  return;
}

template <class FG>
void DiagnosticGraph<FG>::train(const Dataset &dataset) {
  if (!fg_->isValid()) {
    throw std::runtime_error("Factor graph is not valid");
  }
  fg_->train(dataset);
  // Update the weights
  for (auto &[varname, var] : variables_) {
    if (var.type != Variable::Type::kTest) {
      auto prob = fg_->getFailureProbability(var);
      var.prior = prob;
    }
  }
  for (auto &[testname, test] : tests_) {
    auto w = fg_->getWeights(test);
    test.weights = w;
  }
  for (auto &[relname, rel] : relationships_) {
    auto w = fg_->getWeights(rel);
    rel.weights = w;
  }
}

}  // namespace percival
