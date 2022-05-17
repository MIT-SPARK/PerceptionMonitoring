#include <iostream>
#include <string>
#include <vector>

#include "percival/diagnosability/powerset.hpp"

int main() {
  std::vector<std::string> universe = {"a", "b", "c", "d", "e"};
  percival::Powerset<std::string> ps(universe);
  auto result = ps.get(2);
  while (result.has_value()) {
    if (result.value().empty()) {
      std::cout << "EMPTY";
    } else {
      for (const auto& s : result.value()) std::cout << s << " ";
    }
    std::cout << std::endl;
    result = ps.get(2);
  }
  return 0;
}