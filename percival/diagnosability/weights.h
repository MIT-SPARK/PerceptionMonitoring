#pragma once
#include <iostream>
#include <vector>

namespace percival {
class Weights {
 public:
  enum Convention {
    kDensity,
    kEnergy,
  };
  enum Ordering {
    kNatural,
    kReversed,
  };
  Weights(){};
  Weights(const Weights &weights) { weights.get(nat_densities_, Convention::kDensity, Ordering ::kNatural); };
  Weights(const std::vector<double> weights, const Convention convention = Convention::kDensity,
          const Ordering ordering = Ordering::kNatural);
  void randomize();
  std::size_t size() const { return nat_densities_.size(); };
  std::vector<double> get(const Convention convention = Convention::kDensity,
                          const Ordering ordering = Ordering::kNatural) const;
  void get(std::vector<double> &weights, const Convention convention = Convention::kDensity,
           const Ordering ordering = Ordering::kNatural) const;
  static unsigned int ReverseBits(unsigned int num, unsigned int size);
  bool operator==(const Weights &rhs) const { return nat_densities_ == rhs.nat_densities_; };
  bool operator!=(const Weights &rhs) const { return nat_densities_ != rhs.nat_densities_; };
  friend std::ostream &operator<<(std::ostream &os, const Weights &weights);

 private:
  std::vector<double> nat_densities_;
  void reorder_(std::vector<double> &weights, const std::vector<double> original_weights) const;
};
}  // namespace percival