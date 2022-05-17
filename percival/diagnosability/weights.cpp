#include "percival/diagnosability/weights.h"

#include <algorithm>  // std::transform
#include <cmath>
#include <random>

namespace percival {

static constexpr double ToEnergy(double w) {
  const auto epsilon = 1.0e-8;
  if (w <= epsilon)
    return -std::log(epsilon);
  else
    return -std::log(w);
}
static constexpr double ToDensity(double w) { return std::exp(-w); }

Weights::Weights(const std::vector<double> weights, const Convention convention,
                 const Ordering ordering) {
  if (convention == Convention::kDensity) {
    if (ordering == Ordering::kNatural) {
      // Densities in natural order
      nat_densities_ = weights;
    } else {
      // Densities in reversed order
      reorder_(nat_densities_, weights);
    }
  } else {
    if (ordering == Ordering::kNatural) {
      // Energies in natural order
      for (const auto w : weights) nat_densities_.push_back(ToDensity(w));
    } else {
      // Energies in reversed order
      reorder_(nat_densities_, weights);
      std::transform(nat_densities_.begin(), nat_densities_.end(),
                     nat_densities_.begin(), &ToDensity);
    }
  }
}

void Weights::randomize() {
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (auto& w : nat_densities_) w = dist(mt);
}

std::vector<double> Weights::get(const Convention convention,
                                 const Ordering ordering) const {
  std::vector<double> weights;
  if (convention == Convention::kDensity) {
    if (ordering == Ordering::kNatural) {
      // Densities in natural order
      weights = nat_densities_;
    } else {
      // Densities in reversed order
      reorder_(weights, nat_densities_);
    }
  } else {
    if (ordering == Ordering::kNatural) {
      // Energies in natural order
      for (const auto w : nat_densities_) weights.push_back(ToEnergy(w));
    } else {
      // Energies in reversed order
      reorder_(weights, nat_densities_);
      std::transform(weights.begin(), weights.end(), weights.begin(), &ToEnergy);
    }
  }
  return weights;
}

void Weights::get(std::vector<double>& weights, const Convention convention,
                  const Ordering ordering) const {
  if (!weights.empty()) weights.clear();
  if (convention == Convention::kDensity) {
    if (ordering == Ordering::kNatural) {
      // Densities in natural order
      weights = nat_densities_;
    } else {
      // Densities in reversed order
      reorder_(weights, nat_densities_);
    }
  } else {
    if (ordering == Ordering::kNatural) {
      // Energies in natural order
      for (const auto w : nat_densities_) weights.push_back(ToEnergy(w));
    } else {
      // Energies in reversed order
      reorder_(weights, nat_densities_);
      std::transform(weights.begin(), weights.end(), weights.begin(), &ToEnergy);
    }
  }
}

void Weights::reorder_(std::vector<double>& weights,
                       const std::vector<double> original_weights) const {
  if (!weights.empty()) weights.clear();
  auto n = static_cast<unsigned int>(std::log2(original_weights.size()));
  for (unsigned int i = 0; i < original_weights.size(); i++) {
    auto val = original_weights[ReverseBits(i, n)];
    weights.push_back(val);
  }
  return;
}

unsigned int Weights::ReverseBits(unsigned int num, unsigned int size) {
  unsigned int y = 0;
  for (unsigned int i = 0; i < size; ++i) {
    y = (y << 1u) + (1 & num);
    num = num >> 1u;
  }
  return y;
}

std::ostream& operator<<(std::ostream& os, const Weights& weights) {
  os << "[ ";
  for (const auto w : weights.nat_densities_) os << w << " ";
  os << "]";
  return os;
}

}  // namespace percival