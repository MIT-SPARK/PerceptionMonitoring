#include <iostream>
#include <memory>
#include <optional>
#include <vector>

namespace percival {

class BitString {
 public:
  BitString(unsigned int size, unsigned int value) {
    size_ = size;
    value_ = value;
  }

  unsigned int size() const { return size_; }
  unsigned int encoded_permutation() const { return value_; }
  friend std::ostream& operator<<(std::ostream& os, const BitString& bitstring);

  std::vector<bool> to_bool_vector() const {
    std::vector<bool> v(size_, false);
    for (std::size_t i = 0; i < size_; ++i) v[size_ - i - 1] = value_ & (1 << i);
    return v;
  }

  std::vector<unsigned int> to_index_vector() const {
    std::vector<unsigned int> v;
    for (std::size_t i = 0; i < size_; ++i) {
      if (value_ & (1 << i)) v.push_back(i);
    }
    return v;
  }

 private:
  unsigned int size_;
  unsigned int value_;
};

std::ostream& operator<<(std::ostream& os, const BitString& bitstring) {
  os << "[ ";
  for (const auto& b : bitstring.to_bool_vector()) std::cout << b << " ";
  os << "]";
  return os;
}

class LexiBitPerm {
 public:
  LexiBitPerm(unsigned int n, unsigned int t) : n_(n), t_(t), has_next_(true), is_first_(true) {
    // Initialize the permutation
    curr_ = 0;
    for (std::size_t i = 0; i < t; ++i) curr_ = (curr_ << 1) | 1;
    mask_ = 0;
    for (std::size_t i = 0; i < n; ++i) mask_ = (mask_ << 1) | 1;
  };

  BitString last() const { return BitString(n_, curr_); }

  std::optional<BitString> next() {
    if (is_first_) {
      // First call, skip the first permutation computation
      is_first_ = false;
      return BitString(n_, curr_);
    }
    // Assume there are more available permutations
    std::optional<unsigned int> next_perm = {};
    if (has_next_) {
      // Compute the next permutation
      unsigned int t = curr_ | (curr_ - 1);  // t gets curr_'s least significant 0 bits set to 1
      // Next set to 1 the most significant bit to change,
      // set to 0 the least significant ones, and add the necessary 1 bits.
      auto w = (t + 1) | (((~t & -~t) - 1) >> (__builtin_ctz(curr_) + 1));
      // Make sure it was valid (has the correct number of ones)
      const auto ones = __builtin_popcount(w & mask_);
      if (ones == static_cast<int>(t_)) {
        has_next_ = true;  // Assume there are more permutations
        curr_ = w;
        next_perm = w;
      } else {
        has_next_ = false;
      }
    }
    if (next_perm.has_value())
      return BitString(n_, next_perm.value());
    else
      return std::nullopt;
  }

 private:
  unsigned int n_;
  unsigned int t_;
  unsigned int curr_;
  unsigned int mask_;
  bool has_next_;
  bool is_first_;
};

template <typename T>
class Powerset {
 public:
  Powerset(const std::vector<T>& universe) : universe_(universe) {
    const auto n = universe_.size();
    for (std::size_t i = 0; i <= n; ++i) generators_.push_back(std::make_unique<LexiBitPerm>(n, i));
  }

  std::optional<std::vector<T>> get() {
    for (const auto& generator : generators_) {
      auto next = generator->next();
      if (next.has_value()) {
        std::vector<T> result;
        const auto idxs = next.value().to_index_vector();
        for (const auto& i : idxs) result.push_back(universe_[i]);
        return result;
      }
    }
    return std::nullopt;
  }

  std::optional<std::vector<T>> get(std::size_t n) {
    auto next = generators_[n]->next();
    if (next.has_value()) {
      std::vector<T> result;
      const auto idxs = next.value().to_index_vector();
      for (const auto& i : idxs) result.push_back(universe_[i]);
      return result;
    }
    return std::nullopt;
  }

 private:
  std::vector<T> universe_;
  std::vector<std::unique_ptr<LexiBitPerm>> generators_;
};

}  // namespace percival
