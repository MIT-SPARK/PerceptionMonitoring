#pragma once

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/opengm.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/graphicalmodel/graphicalmodel_manipulator.hxx>

#include "percival/diagnosability/typedefs.h"

namespace percival {
namespace factor_graph {
namespace opengm {

typedef double ValueType;           // type used for values
typedef unsigned int LabelType;     // type used for labels (default : size_t)
typedef ::opengm::Multiplier OpType;  // operation used to combine terms
typedef ::opengm::Maximizer OptimizationType; // Multiplier + Maximizer = probability maximization

typedef ::opengm::ExplicitFunction<ValueType, Index, LabelType> ExplicitFunction;  // shortcut for explicite function
// typedef opengm::PottsFunction<ValueType, Index, LabelType> PottsFunction;        // shortcut for Potts function
// typedef opengm::meta::TypeListGenerator<ExplicitFunctionType, PottsFunctionType>::type FunctionTypeList;

typedef ::opengm::SimpleDiscreteSpace<> SpaceType;  // type used to define the feasible statespace

typedef ::opengm::GraphicalModel<ValueType, OpType, ExplicitFunction, SpaceType> Model;  // type of the model
typedef Model::FunctionIdentifier FunctionIdentifier;  // type of the function identifier
typedef ::opengm::GraphicalModelManipulator<Model> ModelManipulator;
typedef ::opengm::GraphicalModelManipulator<Model>::MGM ModifiedModel;

}  // namespace opengm
}  // namespace factor_graph
}  // namespace percival
