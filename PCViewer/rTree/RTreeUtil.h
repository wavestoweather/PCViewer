#pragma once
#include <inttypes.h>
#include <variant>
#include "RTree.h"
#include "RTreeDynamic.h"

namespace RTreeUtil{
    //the RTree union is used to be able to have discrete static rtrees without
    template<class dataType, class elemType>
    using RTreeDimVariant = std::variant<
        RTree<dataType, elemType, 1>,
        RTree<dataType, elemType, 2>,
        RTree<dataType, elemType, 3>,
        RTree<dataType, elemType, 4>,
        RTree<dataType, elemType, 5>,
        RTree<dataType, elemType, 6>,
        RTree<dataType, elemType, 7>,
        RTree<dataType, elemType, 8>,
        RTree<dataType, elemType, 9>,
        RTree<dataType, elemType, 10>,
        RTree<dataType, elemType, 11>,
        RTree<dataType, elemType, 12>,
        RTree<dataType, elemType, 13>,
        RTree<dataType, elemType, 14>,
        RTree<dataType, elemType, 15>,
        RTree<dataType, elemType, 16>,
        RTree<dataType, elemType, 17>,
        RTree<dataType, elemType, 18>,
        RTree<dataType, elemType, 19>,
        RTree<dataType, elemType, 20>,
        RTree<dataType, elemType, 21>,
        RTree<dataType, elemType, 22>,
        RTree<dataType, elemType, 23>,
        RTree<dataType, elemType, 24>,
        RTree<dataType, elemType, 25>,
        RTreeDynamic<dataType, elemType>
    >;
}