#pragma once

#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace percival {

typedef unsigned int Index;
typedef std::string VarName;
typedef std::string Name;
enum FailureModeType { MODULE, OUTPUT };
enum TestOutcome { PASS = 0, FAIL = 1 };
enum FailureModeState { INACTIVE = 0, ACTIVE = 1 };

typedef std::unordered_map<VarName, TestOutcome> Syndrome;
typedef std::unordered_map<VarName, FailureModeState> SystemState;
typedef std::map<Index, TestOutcome, std::less<Index>> TestOutcomeMap;
typedef std::map<Index, FailureModeState, std::less<Index>> FailureModeStateMap;
typedef std::optional<double> FailureProbability;

}  // namespace percival