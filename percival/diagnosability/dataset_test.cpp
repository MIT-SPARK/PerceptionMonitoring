
#include "percival/diagnosability/dataset.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace percival {
namespace utils {
namespace testing {

TEST(DiagnosticGraph_Dataset, Read) {
  percival::Dataset dataset("percival/data/one_diagnosable.train.csv", {"f1", "f2", "f3"}, {"A", "B"});
  EXPECT_EQ(dataset.size(), 8);

  EXPECT_EQ(dataset[0].syndrome, Syndrome({{"A", TestOutcome::PASS}, {"B", TestOutcome::PASS}}));
  EXPECT_EQ(dataset[0].ground_truth, SystemState({{"f1", FailureModeState::INACTIVE},
                                                  {"f2", FailureModeState::INACTIVE},
                                                  {"f3", FailureModeState::INACTIVE}}));
  EXPECT_EQ(dataset[1].syndrome, Syndrome({{"A", TestOutcome::PASS}, {"B", TestOutcome::FAIL}}));
  EXPECT_EQ(dataset[1].ground_truth, SystemState({{"f1", FailureModeState::INACTIVE},
                                                  {"f2", FailureModeState::INACTIVE},
                                                  {"f3", FailureModeState::ACTIVE}}));
  EXPECT_EQ(dataset[2].syndrome, Syndrome({{"A", TestOutcome::FAIL}, {"B", TestOutcome::FAIL}}));
  EXPECT_EQ(dataset[2].ground_truth, SystemState({{"f1", FailureModeState::INACTIVE},
                                                  {"f2", FailureModeState::ACTIVE},
                                                  {"f3", FailureModeState::INACTIVE}}));
  EXPECT_EQ(dataset[3].syndrome, Syndrome({{"A", TestOutcome::FAIL}, {"B", TestOutcome::FAIL}}));
  EXPECT_EQ(dataset[3].ground_truth, SystemState({{"f1", FailureModeState::INACTIVE},
                                                  {"f2", FailureModeState::ACTIVE},
                                                  {"f3", FailureModeState::ACTIVE}}));
  EXPECT_EQ(dataset[4].syndrome, Syndrome({{"A", TestOutcome::FAIL}, {"B", TestOutcome::PASS}}));
  EXPECT_EQ(dataset[4].ground_truth, SystemState({{"f1", FailureModeState::ACTIVE},
                                                  {"f2", FailureModeState::INACTIVE},
                                                  {"f3", FailureModeState::INACTIVE}}));
  EXPECT_EQ(dataset[5].syndrome, Syndrome({{"A", TestOutcome::FAIL}, {"B", TestOutcome::FAIL}}));
  EXPECT_EQ(dataset[5].ground_truth, SystemState({{"f1", FailureModeState::ACTIVE},
                                                  {"f2", FailureModeState::INACTIVE},
                                                  {"f3", FailureModeState::ACTIVE}}));
  EXPECT_EQ(dataset[6].syndrome, Syndrome({{"A", TestOutcome::FAIL}, {"B", TestOutcome::FAIL}}));
  EXPECT_EQ(dataset[6].ground_truth, SystemState({{"f1", FailureModeState::ACTIVE},
                                                  {"f2", FailureModeState::ACTIVE},
                                                  {"f3", FailureModeState::INACTIVE}}));
  EXPECT_EQ(dataset[7].syndrome, Syndrome({{"A", TestOutcome::FAIL}, {"B", TestOutcome::FAIL}}));
  EXPECT_EQ(dataset[7].ground_truth, SystemState({{"f1", FailureModeState::ACTIVE},
                                                  {"f2", FailureModeState::ACTIVE},
                                                  {"f3", FailureModeState::ACTIVE}}));
}

}  // namespace testing
}  // namespace utils
}  // namespace percival