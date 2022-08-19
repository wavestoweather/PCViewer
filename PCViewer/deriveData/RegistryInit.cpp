#include "Nodes.hpp"

using namespace deriveData;
std::map<std::string, NodesRegistry::Entry> NodesRegistry::nodes{};
REGISTER_NODE(DatasetInputNode);
REGISTER_NODE(ZeroVectorNode);
REGISTER_NODE(OneVectorNode);
REGISTER_NODE(RandomVectorNode);
REGISTER_NODE(PrintVectorNode);
REGISTER_NODE(MultiplicationInverseNode);
REGISTER_NODE(AdditionInverseNode);
REGISTER_NODE(NormalizationNode);
REGISTER_NODE(AbsoluteValueNode);
REGISTER_NODE(SquareNode);
REGISTER_NODE(ExponentialNode);
REGISTER_NODE(LogarithmNode);
REGISTER_NODE(CreateVec2Node);
REGISTER_NODE(SplitVec2);
REGISTER_NODE(Vec2Norm);
REGISTER_NODE(CreateVec3Node);
REGISTER_NODE(SplitVec3);
REGISTER_NODE(Vec3Norm);
REGISTER_NODE(CreateVec4Node);
REGISTER_NODE(SplitVec4);
REGISTER_NODE(Vec4Norm);
REGISTER_NODE(PlusNode);
REGISTER_NODE(MinusNode);
REGISTER_NODE(MultiplicationNode);
REGISTER_NODE(DivisionNode);
