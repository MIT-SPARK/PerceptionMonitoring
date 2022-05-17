#include "percival/diagnosability/utils.h"

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "percival/diagnosability/typedefs.h"
#include "percival/diagnosability/weights.h"

namespace percival {
namespace utils {

TEST(DiagnosticGraph_Utils, WeightsGenerator) {
  std::vector<VarName> scope = {"f1", "f2"};
  WeightsGeneratorFunction f = [](const SystemState &state) -> double {
    return (state.at("f1")) ? 1.0 : 0.0;
  };
  auto weights = WeightsGenerator(scope, f);
  std::vector<double> w;
  weights.get(w);
  ASSERT_THAT(w, testing::ElementsAre(0.0, 0.0, 1.0, 1.0));
}

TEST(DiagnosticGraph_Utils, TestWeightsGenerator_PerfectTest) {
  percival::Test t(VarName("t0"), VarName("t0"), 0, {VarName("f1"), VarName("f2")}, {});
  auto weights = TestWeightsGenerator(t, &PerfectTest);
  std::vector<double> w;
  weights.get(w);
  ASSERT_THAT(w, testing::ElementsAre(1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0));
}

TEST(DiagnosticGraph_Utils, TestWeightsGenerator_IdealTest) {
  percival::Test t(VarName("t0"), VarName("t0"), 0, {VarName("f1"), VarName("f2")}, {});
  auto weights = TestWeightsGenerator(t, &IdealTest);
  std::vector<double> w;
  weights.get(w);
  ASSERT_THAT(w, testing::ElementsAre(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
}

TEST(DiagnosticGraph_Utils, TestWeightsGenerator_AllFail) {
  std::vector<VarName> scope = {"f1", "f2"};
  auto weights = WeightsGenerator(scope, &AllFailOrNone);
  std::vector<double> w;
  weights.get(w);
  ASSERT_THAT(w, testing::ElementsAre(1.0, 0.0, 0.0, 1.0));
}

    
}  // namespace utils
}  // namespace percival
