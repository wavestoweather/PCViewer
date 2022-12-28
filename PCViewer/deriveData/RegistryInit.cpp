#include "Nodes.hpp"

using namespace deriveData::Nodes;
std::map<std::string, Registry::Entry> Registry::nodes{};
REGISTER(DatasetInput);
REGISTER(VectorZero);
REGISTER(VectorOne);
REGISTER(VectorRandom);
REGISTER(PrintVector);
REGISTER(DatasetOutput);
REGISTER(Derivation);
REGISTER(Sum);
REGISTER(Norm);
REGISTER(PCA_Projection);
REGISTER(TSNE_Projection);
// unary nodes
REGISTER(Inverse);
REGISTER(Negate);
REGISTER(Normalization);
REGISTER(AbsoluteValue);
REGISTER(Square);
REGISTER(Sqrt);
REGISTER(Exponential);
REGISTER(Logarithm);
//REGISTER(CreateVec2);
//REGISTER(SplitVec2);
//REGISTER(Vec2Norm);
//REGISTER(CreateVec3);
//REGISTER(SplitVec3);
//REGISTER(Vec3Norm);
//REGISTER(CreateVec4);
//REGISTER(SplitVec4);
//REGISTER(Vec4Norm);
// binary nodes
REGISTER(Plus);
REGISTER(Minus);
REGISTER(Multiplication);
REGISTER(Division);
REGISTER(Pow);
REGISTER(Min_Reduction);
REGISTER(Max_Reduction);
REGISTER(Sum_Reduction);
REGISTER(Mul_Reduction);
REGISTER(Average_Reduction);
REGISTER(StdDev_Reduction);
//REGISTER(PlusVec2);
//REGISTER(MinusVec2);
//REGISTER(MultiplicationVec2);
//REGISTER(DivisionVec2);
