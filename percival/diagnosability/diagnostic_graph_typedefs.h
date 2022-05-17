#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "percival/diagnosability/typedefs.h"
#include "percival/diagnosability/weights.h"

namespace percival {

struct Variable {
  VarName name;
  Index index;
  enum Type { kModuleFailureMode, kOutputFailureMode, kTest } type;
  FailureProbability prior;
  Variable(){};
  Variable(const VarName &name, const Index index, const Type type,
           const FailureProbability prior = {})
      : name(name), index(index), type(type), prior(prior){};
};

struct Test {
  Name name;        // nane of the test itself
  VarName varname;  // unique identifier for the test outcome
  Index index;
  std::vector<VarName> scope;
  Weights weights;
  Test() {}
  Test(const Name &name, const VarName &varname, const Index index,
       const std::vector<VarName> &scope, const Weights &weights)
      : name(name), varname(varname), index(index), scope(scope), weights(weights) {}
};

struct Relationship {
  Name name;  // name of the relationship
  VarName varname; // unique identifier for the relationship
  std::vector<VarName> scope;
  Weights weights;
  Relationship() {}
  Relationship(const Name &name, const VarName &varname,
               const std::vector<VarName> &scope, const Weights &weights)
      : name(name), varname(varname), scope(scope), weights(weights) {}
};
}  // namespace percival