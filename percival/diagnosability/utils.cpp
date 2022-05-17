#include "percival/diagnosability/utils.h"

#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace percival {
namespace utils {

unsigned int NumActiveFailures(const SystemState &system_state) {
  unsigned int num_active_failures = 0;
  for (const auto &s : system_state)
    num_active_failures += static_cast<unsigned int>(s.second);
  return num_active_failures;
}

Weights TestWeightsGenerator(const Test &test, TestWeightsGeneratorFunction gen) {
  const unsigned int card = test.scope.size();
  const unsigned int table_cardinality = 1u << (card + 1);
  std::vector<double> w(table_cardinality);
  for (unsigned int x = 0; x < table_cardinality; ++x) {
    SystemState state;
    const TestOutcome t = static_cast<TestOutcome>(((1u << card) & x) >> card);
    for (unsigned int c = 0; c < card; ++c) {
      const VarName v = test.scope[card - c - 1];
      state[v] = static_cast<FailureModeState>(((1u << c) & x) >> c);
    }
    w[x] = gen(t, state);
  }
  return Weights(w);
}

Weights WeightsGenerator(const std::vector<VarName> &scope,
                         WeightsGeneratorFunction gen) {
  const unsigned int card = scope.size();
  const unsigned int table_cardinality = 1u << (card);
  std::vector<double> w(table_cardinality);
  for (unsigned int x = 0; x < table_cardinality; ++x) {
    SystemState state;
    for (unsigned int c = 0; c < card; ++c) {
      const VarName v = scope[card - c - 1];
      state[v] = static_cast<FailureModeState>(((1u << c) & x) >> c);
    }
    w[x] = gen(state);
  }

  return Weights(w);
}

double PerfectTest(const TestOutcome t, const SystemState &scope_state) {
  const unsigned int card = scope_state.size();
  unsigned int num_failures = 0;
  for (const auto &s : scope_state) num_failures += static_cast<unsigned int>(s.second);
  if (num_failures == card)
    return 1.0; // Everithing is possible
  else if (num_failures == 0)
    return static_cast<double>(t == TestOutcome::PASS); // only PASS is feasible
  else
    return static_cast<double>(t == TestOutcome::FAIL); // only FAIL is feasible
}

double IdealTest(const TestOutcome t, const SystemState &scope_state) {
  unsigned int num_failures = 0;
  for (const auto &s : scope_state) num_failures += static_cast<unsigned int>(s.second);
  if (num_failures > 0) {
    if (t == TestOutcome::FAIL)
      return 1.0;
    else
      return 0.0;
  } else {
    if (t == TestOutcome::FAIL)
      return 0.0;
    else
      return 1.0;
  }
}

double AllFailOrNone(const SystemState &scope_state) {
  unsigned int num_failures = 0;
  for (const auto &s : scope_state) num_failures += static_cast<unsigned int>(s.second);
  if (num_failures == 0 or num_failures == scope_state.size()) {
    return 1.0;
  } else {
    return 0.0;
  }
}

double RandomTest(const TestOutcome t, const SystemState &scope_state) {
  std::random_device rd;   // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<short> distrib(0, 1);
  return static_cast<double>(distrib(gen));
}

double Random(const SystemState &scope_state) {
  std::random_device rd;   // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<short> distrib(0, 1);
  return static_cast<double>(distrib(gen));
}

}  // namespace utils
}  // namespace percival