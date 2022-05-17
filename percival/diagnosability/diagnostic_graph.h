#pragma once
#include <memory>
#include <optional>
#include <string>
#include <map>
#include <vector>

#include "percival/diagnosability/dataset.h"
#include "percival/diagnosability/diagnostic_graph_typedefs.h"
namespace percival {

template <class FG>
class DiagnosticGraph {
 public:
  DiagnosticGraph();

  template <typename T>
  DiagnosticGraph<FG>(const DiagnosticGraph<T> &rhs);

  void addFailureMode(const VarName &name, const FailureModeType &type,
                      const FailureProbability prior = {});
  void addTest(const Name &name, const VarName &varname,
               const std::vector<VarName> &scope, const std::vector<double> &weights);
  void addTest(const Name &name, const VarName &varname,
               const std::vector<VarName> &scope,
               double (*gen)(const TestOutcome, const SystemState &));
  void addRelationship(const Name &name, const VarName &varname,
                       const std::vector<VarName> &scope,
                       const std::vector<double> &weights);
  void addRelationship(const Name &name, const VarName &varname,
                       const std::vector<VarName> &scope,
                       double (*gen)(const SystemState &));

  void randomizeWeights();

  void fromJson(const std::string &filepath);
  void toJson(const std::string &filepath) const;

  std::shared_ptr<FG> factor_graph() const { return fg_; };
  const std::map<VarName, Variable> &variables() const { return variables_; };
  const std::map<VarName, Test> &tests() const { return tests_; };
  const std::map<VarName, Relationship> &relationships() const {
    return relationships_;
  };
  const std::optional<unsigned int> max_failure_cardinality() const;
  void max_failure_cardinality(unsigned int card);
  const std::vector<VarName> failure_mode_names() const;
  const std::vector<VarName> test_names() const;
  const std::size_t num_failure_modes() const;
  const std::size_t num_tests() const { return tests_.size(); };
  std::optional<const Variable *> getVariable(const Index id) const;
  std::optional<const Variable *> getVariable(const VarName &name) const;

  void bake() { fg_->bake(this); };
  void load(const std::string &filepath) { fg_->load(this, filepath); };
  void print() const;
  void summary() const;

  SystemState faultIdentification(const Syndrome &syn) const {
    return fg_->faultIdentification(syn);
  };
  SystemState marginalEstimation(const SystemState &state) const {
    return fg_->marginalEstimation(state);
  };
  void train(const Dataset &dataset);

 private:
  std::shared_ptr<FG> fg_;
  std::map<VarName, Variable> variables_;
  std::map<VarName, Test> tests_;
  std::map<VarName, Relationship> relationships_;
  std::optional<unsigned int> max_failure_cardinality_;

  void addTest_(const Test &test);
  void addRelationship_(const Relationship &relationship);
};

template <class FG>
using DiagnosticGraphPtr = std::shared_ptr<DiagnosticGraph<FG>>;

}  // namespace percival

#include "percival/diagnosability/diagnostic_graph.tpp"