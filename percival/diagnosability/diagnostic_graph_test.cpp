#include "percival/diagnosability/diagnostic_graph.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "percival/diagnosability/dataset.h"
#include "percival/diagnosability/example_graphs.hpp"
#include "percival/diagnosability/factor_graph/dummy.h"
#include "percival/diagnosability/typedefs.h"
#include "percival/diagnosability/utils.h"

namespace percival {
namespace testing {

DiagnosticGraph<factor_graph::dummy::FactorGraph> SimpleGraph(bool with_modules = false,
                                                              bool randmomize = false) {
  DiagnosticGraph<factor_graph::dummy::FactorGraph> dgraph;
  if (with_modules) {
    dgraph.addFailureMode("m1", FailureModeType::MODULE, 0.1);
    dgraph.addFailureMode("m2", FailureModeType::MODULE, 0.1);
  }

  dgraph.addFailureMode("f1", FailureModeType::OUTPUT);
  dgraph.addFailureMode("f2", FailureModeType::OUTPUT);

  if (randmomize)
    dgraph.addTest("A", "A", {"f1", "f2"}, &utils::RandomTest);
  else
    dgraph.addTest("A", "A", {"f1", "f2"}, &utils::PerfectTest);

  if (with_modules) {
    if (randmomize) {
      dgraph.addRelationship("r1", "r1", {"m1", "f1"}, &utils::Random);
      dgraph.addRelationship("r2", "r1", {"m2", "f2"}, &utils::Random);
    } else {
      dgraph.addRelationship("r1", "r1", {"m1", "f1"}, &utils::AllFailOrNone);
      dgraph.addRelationship("r2", "r2", {"m2", "f2"}, &utils::AllFailOrNone);
    }
  }
  return dgraph;
}

TEST(DiagnosticGraph, Simple) {
  DiagnosticGraph<factor_graph::dummy::FactorGraph> dgraph;
  dgraph.addFailureMode("f1", FailureModeType::OUTPUT);
  dgraph.addFailureMode("f2", FailureModeType::OUTPUT);
  dgraph.addFailureMode("f3", FailureModeType::OUTPUT);
  dgraph.addTest("A", "A", {"f1", "f2"}, &utils::PerfectTest);
  dgraph.addTest("B", "B", {"f2", "f3"}, &utils::PerfectTest);
  EXPECT_EQ(dgraph.num_failure_modes(), 3);
  EXPECT_EQ(dgraph.num_tests(), 2);
}

TEST(DiagnosticGraph, FromJSON) {
  DiagnosticGraph<factor_graph::dummy::FactorGraph> dgraph;
  dgraph.fromJson("percival/data/dummy.json");
  EXPECT_EQ(dgraph.num_failure_modes(), 6);
  EXPECT_EQ(dgraph.num_tests(), 2);
}

TEST(DiagnosticGraph, ToJSON) {
  auto dgraph = SimpleGraph();
  dgraph.toJson("json_test.json");

  DiagnosticGraph<factor_graph::dummy::FactorGraph> dgraph2;
  dgraph2.fromJson("json_test.json");

  ASSERT_EQ(dgraph2.num_failure_modes(), 2);
  ASSERT_EQ(dgraph2.num_tests(), 1);
}

TEST(DiagnosticGraph, Inference) {
  auto dgraph = SimpleGraph(false, true);
  dgraph.bake();
  {
    Syndrome syn = {{"A", TestOutcome::PASS}};
    auto fi = dgraph.faultIdentification(syn);
    SystemState expected_fi = {{"f1", FailureModeState::INACTIVE},
                               {"f2", FailureModeState::INACTIVE}};
    EXPECT_EQ(fi, expected_fi);
  }
  {
    Syndrome syn = {{"A", TestOutcome::FAIL}};
    auto fi = dgraph.faultIdentification(syn);
    SystemState expected_fi = {{"f1", FailureModeState::ACTIVE},
                               {"f2", FailureModeState::ACTIVE}};
    EXPECT_EQ(fi, expected_fi);
  }
}

TEST(DiagnosticGraph, Training) {
  auto dgraph = example::OneDiagnosable<factor_graph::dummy::FactorGraph>();
  ASSERT_THAT(dgraph->tests().at("A").weights.get(),
              ::testing::ElementsAre(1, 0, 0, 1, 0, 1, 1, 1));
  ASSERT_THAT(dgraph->tests().at("B").weights.get(),
              ::testing::ElementsAre(1, 0, 0, 1, 0, 1, 1, 1));
  dgraph->bake();
  Dataset dataset("percival/data/one_diagnosable.train.csv",
                  dgraph->failure_mode_names(), dgraph->test_names());
  dgraph->train(dataset);
  ASSERT_THAT(dgraph->tests().at("A").weights.get(),
              ::testing::ElementsAre(1, 0, 0, 0, 0, 1, 1, 1));
  ASSERT_THAT(dgraph->tests().at("B").weights.get(),
              ::testing::ElementsAre(1, 0, 0, 0, 0, 1, 1, 1));
}

}  // namespace testing
}  // namespace percival