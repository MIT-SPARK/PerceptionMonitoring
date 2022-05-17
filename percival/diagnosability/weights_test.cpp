#include "percival/diagnosability/weights.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace percival {
namespace testing {

TEST(Weights, Comparison) {
  Weights a({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}, Weights::Convention::kDensity, Weights::Ordering::kNatural);
  Weights b({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}, Weights::Convention::kDensity, Weights::Ordering::kNatural);
  Weights c(
      {
          0.0,
          1.0,
          2.0,
          3.0,
      },
      Weights::Convention::kDensity, Weights::Ordering::kNatural);
  EXPECT_EQ(a, b);
  EXPECT_NE(a, c);
}

TEST(Weights, Weights) {
  std::vector<double> nat_den = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
  Weights w(nat_den, Weights::Convention::kDensity, Weights::Ordering::kNatural);

  {
    std::vector<double> x;
    w.get(x, Weights::Convention::kDensity, Weights::Ordering::kNatural);
    ASSERT_THAT(x, ::testing::ElementsAre(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0));
  }
  {
    std::vector<double> x;
    w.get(x, Weights::Convention::kDensity, Weights::Ordering::kReversed);
    ASSERT_THAT(x, ::testing::ElementsAre(0.0, 4.0, 2.0, 6.0, 1.0, 5.0, 3.0, 7.0));
  }
  {
    std::vector<double> x;
    w.get(x, Weights::Convention::kEnergy, Weights::Ordering::kNatural);
    ASSERT_THAT(x, ::testing::ElementsAre(0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0));
  }
  {
    std::vector<double> x;
    w.get(x, Weights::Convention::kEnergy, Weights::Ordering::kReversed);
    ASSERT_THAT(x, ::testing::ElementsAre(0.0, -4.0, -2.0, -6.0, -1.0, -5.0, -3.0, -7.0));
  }
}

}  // namespace testing
}  // namespace percival
