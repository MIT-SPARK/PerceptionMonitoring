#pragma once

#include <functional>

#include "percival/diagnosability/diagnostic_graph_typedefs.h"
#include "percival/diagnosability/weights.h"

namespace percival {
namespace utils {

typedef std::function<double(const TestOutcome, const SystemState &)>
    TestWeightsGeneratorFunction;
typedef std::function<double(const SystemState &)> WeightsGeneratorFunction;

unsigned int NumActiveFailures(const SystemState &system_state);
Weights TestWeightsGenerator(const Test &test,
                             TestWeightsGeneratorFunction gen);
Weights WeightsGenerator(const std::vector<VarName> &scope,
                         WeightsGeneratorFunction gen);
double PerfectTest(const TestOutcome t, const SystemState &scope_state);
double IdealTest(const TestOutcome t, const SystemState &scope_state);
double RandomTest(const TestOutcome t, const SystemState &scope_state);
double AllFailOrNone(const SystemState &scope_state);
double Random(const SystemState &scope_state);

}  // namespace utils
}  // namespace percival