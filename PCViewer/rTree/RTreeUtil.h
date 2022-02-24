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

    template<class dataType, class elemType>
    class RTreeAPI{
    public:
        virtual int Search(const elemType* a_min, const elemType* a_max, std::function<bool (const dataType&)> callback) const = 0;
        virtual void Remove(const elemType* a_min, const elemType* a_max, const dataType& a_dataId) = 0;
        virtual void Insert(const elemType* a_min, const elemType* a_max, const dataType& a_dataId) = 0;
        virtual void RemoveAll() = 0;
        virtual int NumDims() const = 0;
        virtual uint32_t ByteSize() const = 0;
    };

    template<class dataType, class elemType, int numDims>
    class RTreeS: public RTreeAPI<dataType, elemType>, public RTree<dataType, elemType, numDims>{
    public:
        int Search(const elemType* a_min, const elemType* a_max, std::function<bool (const dataType&)> callback) const{return RTree<dataType, elemType, numDims>::Search(a_max, a_min, callback);};
        void Remove(const elemType* a_min, const elemType* a_max, const dataType& a_dataId){RTree<dataType, elemType, numDims>::Remove(a_min, a_max, a_dataId);};
        void Insert(const elemType* a_min, const elemType* a_max, const dataType& a_dataId){RTree<dataType, elemType, numDims>::Insert(a_min, a_max, a_dataId);};
        void RemoveAll() {RTree<dataType, elemType, numDims>::RemoveAll();};
        int NumDims() const {return numDims;};
        uint32_t ByteSize() const{return RTree<dataType, elemType, numDims>::BYTE_SIZE;};
    };

    template<class dataType, class elemType>
    class RTreeD: public RTreeAPI<dataType, elemType>, public RTreeDynamic<dataType, elemType>{
    public:
        RTreeD(int numDims):RTreeDynamic<dataType, elemType>::RTreeDynamic<dataType, elemType>(numDims){};
        int Search(const elemType* a_min, const elemType* a_max, std::function<bool (const dataType&)> callback) const{return RTreeDynamic<dataType, elemType>::Search(a_max, a_min, callback);};
        void Remove(const elemType* a_min, const elemType* a_max, const dataType& a_dataId){RTreeDynamic<dataType, elemType>::Remove(a_min, a_max, a_dataId);};
        void Insert(const elemType* a_min, const elemType* a_max, const dataType& a_dataId){RTreeDynamic<dataType, elemType>::Insert(a_min, a_max, a_dataId);};
        void RemoveAll() {RTreeDynamic<dataType, elemType>::RemoveAll();};
        int NumDims() const{return RTreeDynamic<dataType, elemType>::NUMDIMS;};
        uint32_t ByteSize() const{return RTreeDynamic<dataType, elemType>::BYTE_SIZE;};
    };
}