#include "percival/diagnosability/dataset.h"

#include "csv.hpp"

namespace percival {

Dataset::Dataset(const std::string& filename, const std::vector<VarName>& failure_modes,
                 const std::vector<VarName>& tests) {
  csv::CSVReader reader(filename);
  for (auto& row : reader) {
    auto ground_truth = SystemState();
    for (const auto& fm : failure_modes) {
      try {
        const auto value = row[fm].get<unsigned int>();
        ground_truth[fm] = static_cast<FailureModeState>(value);
      } catch (std::runtime_error& e) {
        // If any of the FailureModeStates are not available, skip entirely
        ground_truth = SystemState();
        break;
      }
    }
    auto syndrome = Syndrome();
    for (const auto& t : tests) {
      const auto value = row[t].get<unsigned int>();
      syndrome[t] = static_cast<TestOutcome>(value);
    }
    samples_.push_back(Sample(syndrome, ground_truth));
  }
}

}  // namespace percival
