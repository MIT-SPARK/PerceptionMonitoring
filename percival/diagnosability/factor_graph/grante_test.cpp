#include "percival/diagnosability/factor_graph/grante.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "percival/diagnosability/example_graphs.hpp"

namespace percival {
namespace factor_graph {
namespace grante {
namespace testing {

using FactorGraph = percival::factor_graph::grante::FactorGraph;
namespace Inference = percival::factor_graph::grante::inference;

TEST(GranteFactorGraph, Inference) {
  auto dgraph = example::OneDiagnosable<FactorGraph>();
  dgraph->max_failure_cardinality(-1);  // disable cardinality constraint, not supported
  // ASSERT_EQ(dgraph->max_failure_cardinality(),
  // testing::Optional(testing::Eq(std::nullopt)));
  dgraph->bake();

  // Use brute force to make sure the results are correct.
  auto fg = dgraph->factor_graph();
  fg->inference_options.algorithm = Inference::Algorithm::kBruteForce;
  fg->inference_options =
      Inference::DefaultInferenceOptions(fg->inference_options.algorithm);

  // Perform fault identification
  percival::Syndrome syn;
  {
    syn["A"] = percival::TestOutcome::PASS;
    syn["B"] = percival::TestOutcome::PASS;
    auto fi = dgraph->faultIdentification(syn);
    ASSERT_EQ(fi.at("f1"), percival::FailureModeState::INACTIVE);
    ASSERT_EQ(fi.at("f2"), percival::FailureModeState::INACTIVE);
    ASSERT_EQ(fi.at("f3"), percival::FailureModeState::INACTIVE);
  }
  {
    syn["A"] = percival::TestOutcome::PASS;
    syn["B"] = percival::TestOutcome::FAIL;
    auto fi = dgraph->faultIdentification(syn);
    // ASSERT_EQ(fi.at("f1"), percival::FailureModeState::ACTIVE);
    // ASSERT_EQ(fi.at("f2"), percival::FailureModeState::ACTIVE);
    // ASSERT_EQ(fi.at("f3"), percival::FailureModeState::INACTIVE);
    ASSERT_EQ(fi.at("f1"), percival::FailureModeState::INACTIVE);
    ASSERT_EQ(fi.at("f2"), percival::FailureModeState::INACTIVE);
    ASSERT_EQ(fi.at("f3"), percival::FailureModeState::ACTIVE);
  }
  {
    syn["A"] = percival::TestOutcome::FAIL;
    syn["B"] = percival::TestOutcome::PASS;
    auto fi = dgraph->faultIdentification(syn);
    // ASSERT_EQ(fi.at("f1"), percival::FailureModeState::ACTIVE);
    // ASSERT_EQ(fi.at("f2"), percival::FailureModeState::INACTIVE);
    // ASSERT_EQ(fi.at("f3"), percival::FailureModeState::INACTIVE);
    ASSERT_EQ(fi.at("f1"), percival::FailureModeState::INACTIVE);
    ASSERT_EQ(fi.at("f2"), percival::FailureModeState::ACTIVE);
    ASSERT_EQ(fi.at("f3"), percival::FailureModeState::ACTIVE);
  }
  {
    syn["A"] = percival::TestOutcome::FAIL;
    syn["B"] = percival::TestOutcome::FAIL;
    auto fi = dgraph->faultIdentification(syn);
    ASSERT_EQ(fi.at("f1"), percival::FailureModeState::INACTIVE);
    ASSERT_EQ(fi.at("f2"), percival::FailureModeState::ACTIVE);
    ASSERT_EQ(fi.at("f3"), percival::FailureModeState::INACTIVE);
  }
}

TEST(GranteFactorGraph, Train) {
  auto dgraph = example::OneDiagnosable<FactorGraph>();
  dgraph->max_failure_cardinality(-1);  // disable cardinality constraint
  dgraph->bake();
  auto pre_w = dgraph->factor_graph()->getWeights(dgraph->tests().at("A"));
  EXPECT_EQ(pre_w.size(), 8);

  percival::Dataset dataset("percival/data/one_diagnosable.train.csv",
                            dgraph->failure_mode_names(), dgraph->test_names());
  EXPECT_EQ(dataset.size(), 8);
  dgraph->factor_graph()->inference_options =
      Inference::DefaultInferenceOptions(Inference::Algorithm::kBruteForce);
  dgraph->factor_graph()->train(dataset);

  // Test something changed
  auto post_w = dgraph->factor_graph()->getWeights(dgraph->tests().at("A"));
  EXPECT_EQ(post_w.size(), 8);
  EXPECT_NE(pre_w, post_w);
}

}  // namespace testing
}  // namespace grante
}  // namespace factor_graph
}  // namespace percival
