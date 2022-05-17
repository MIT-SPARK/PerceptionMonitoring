#pragma once

#include <optional>
#include <stdexcept>
#include <vector>

#include "percival/diagnosability/typedefs.h"

namespace percival {

struct Sample {
  Syndrome syndrome;
  SystemState ground_truth;

  Sample(){};
  Sample(const Syndrome &syndrome, const SystemState &ground_truth)
      : syndrome(syndrome), ground_truth(ground_truth){};

  std::optional<unsigned int> at(const VarName &name) const {
    for (const auto &[key, value] : ground_truth) {
      if (key == name) return value;
    }
    for (const auto &[key, value] : syndrome) {
      if (key == name) return value;
    }
    return std::nullopt;
  }
};

class Dataset {
 public:
  typedef typename std::vector<Sample> Samples;
  typedef typename Samples::iterator iterator;
  typedef typename Samples::const_iterator const_iterator;

  Dataset(const std::string &filename, const std::vector<VarName> &failure_modes,
          const std::vector<VarName> &tests);
  std::size_t size() const { return samples_.size(); };

  Sample operator[](std::size_t idx) const { return samples_.at(idx); }
  Sample &operator[](std::size_t idx) { return samples_.at(idx); }

  inline iterator begin() noexcept { return samples_.begin(); };
  inline iterator end() noexcept { return samples_.end(); };
  inline const_iterator begin() const noexcept { return samples_.begin(); };
  inline const_iterator end() const noexcept { return samples_.end(); };
  inline const_iterator cbegin() const noexcept { return samples_.cbegin(); };
  inline const_iterator cend() const noexcept { return samples_.cend(); };

 private:
  std::vector<Sample> samples_;
};

}  // namespace percival