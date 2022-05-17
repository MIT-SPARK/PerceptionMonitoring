#pragma once

#include <bitset>
#include <initializer_list>

#include "percival/diagnosability/dataset.h"
#include "percival/diagnosability/diagnostic_graph.h"
#include "percival/diagnosability/typedefs.h"
#include "percival/diagnosability/weights.h"

namespace percival {
namespace factor_graph {

class Capabilities {
 public:
  enum Capability { INFERENCE = 0, TRAINING = 1, MAX_CARDINALITY = 2 };

  Capabilities() = default;
  Capabilities(std::initializer_list<Capability> caps) {
    for (const auto c : caps) this->set(c);
  }

  void set(Capability t) { capabilities_.set(static_cast<std::size_t>(t)); }
  void reset(Capability t) { capabilities_.reset(static_cast<std::size_t>(t)); };
  bool query(Capability t) const { return capabilities_[static_cast<std::size_t>(t)]; };
  bool operator()(Capability t) const { return this->query(t); }

 private:
  std::bitset<3> capabilities_;
};

class FactorGraphBase {
 public:
  virtual bool isValid() const = 0;
  virtual SystemState faultIdentification(const Syndrome &syn) const = 0;
  virtual void train(const Dataset &dataset) = 0;
  virtual void print() const = 0;

  virtual FailureProbability getFailureProbability(const Variable &var) const = 0;
  virtual Weights getWeights(const Test &test) const = 0;
  virtual Weights getWeights(const Relationship &relationship) const = 0;

  virtual const Capabilities getCapabilities() const = 0;
};

}  // namespace factor_graph
}  // namespace percival