
#include <cmath>
using std::abs;
#include <list>
using std::list;
#include <vector>
using std::vector;
#include <utility>
using std::pair;
#include <boost/shared_ptr.hpp>
using boost::shared_ptr;
using boost::static_pointer_cast;
#include <cocotte/models/leaf.h>
#include <cocotte/models/node.h>
#include <cocotte/models/modeliterator.h>


namespace Cocotte {
namespace Models {



// Constructors
template <typename ModelType, typename PointType>
ModelIt<ModelType, PointType>::ModelIt(list<shared_ptr<ModelType>> t, int d) : treeBranch(t), depth(d)
{}


template <typename ModelType, typename PointType>
ModelIt<ModelType, PointType>::ModelIt(ModelIt const& mIt) : treeBranch(mIt.treeBranch), depth(mIt.depth)
{}



// Accessors
template <typename ModelType, typename PointType>
list<shared_ptr<Model>> const& ModelIt<ModelType, PointType>::getTreeBranch() const
{
    return treeBranch;
}


template <typename ModelType, typename PointType>
int ModelIt<ModelType, PointType>::getDepth() const
{
    return depth;
}



// Operators
template <typename ModelType, typename PointType>
shared_ptr<PointType const> ModelIt<ModelType, PointType>::getSharedPointer()
{
    shared_ptr<LeafType> const leaf = static_pointer_cast<LeafType>(treeBranch.back());
    return leaf->getPointAddress();
}


template <typename ModelType, typename PointType>
PointType const& ModelIt<ModelType, PointType>::operator*()
{
    shared_ptr<LeafType> const leaf = static_pointer_cast<LeafType>(treeBranch.back());
    return leaf->getPoint();
}


template <typename ModelType, typename PointType>
PointType const* ModelIt<ModelType, PointType>::operator->()
{
    shared_ptr<LeafType> const leaf = static_pointer_cast<LeafType>(treeBranch.back());
    return &(leaf->getPoint());
}


template <typename ModelType, typename PointType>
ModelIt<ModelType, PointType>& ModelIt<ModelType, PointType>::operator++()
{
    ModelType* lastVisited = treeBranch.back().get();
    treeBranch.pop_back();
    --depth;

    if (treeBranch.empty())
    {
        return *this;
    }

    shared_ptr<NodeType> currentNode = static_pointer_cast<NodeType>(treeBranch.back());

    while (currentNode->getModel1().get() == lastVisited)
    {
        lastVisited = currentNode.get();
        treeBranch.pop_back();
        --depth;

        if (treeBranch.empty())
        {
            return *this;
        }

        currentNode = static_pointer_cast<NodeType>(treeBranch.back());
    }

    shared_ptr<ModelType> newlyVisited = currentNode->getModel1();
    treeBranch.push_back(newlyVisited);
    ++depth;

    while (!newlyVisited->isLeaf())
    {
        newlyVisited = static_pointer_cast<NodeType>(newlyVisited)->getModel0();
        treeBranch.push_back(newlyVisited);
        ++depth;
    }

    return *this;
}


template <typename ModelType, typename PointType>
ModelIt<ModelType, PointType> ModelIt<ModelType, PointType>::operator++(int)
{
    ModelIt<ModelType, PointType> tmp(*this);
    operator++();
    return tmp;
}


template <typename ModelType, typename PointType>
bool ModelIt<ModelType, PointType>::operator==(ModelIt const& rhs)
{
    if (depth != rhs.depth)
    {
        return false;
    }

    auto const pIt = treeBranch.begin();
    for (auto const& m : rhs.treeBranch)
    {
        if(m.get() != pIt->get())
        {
            return false;
        }
    }

    return true;
}


template <typename ModelType, typename PointType>
bool ModelIt<ModelType, PointType>::operator!=(const ModelIt& rhs)
{
    return !operator==(rhs);
}


// Iterators
template<typename ModelType,
         typename IteratorType = typename std::conditional<std::is_const<ModelType>::value, ModelConstIterator, ModelIterator>::type,
         typename = typename std::enable_if<std::is_same<Model, typename std::decay<ModelType>::type>::value>::type>
IteratorType pointsBegin(boost::shared_ptr<ModelType> pModel)
{
    list<shared_ptr<ModelType>> ptrList(1, pModel);
    int depth = 1;

    while (!pModel->isLeaf())
    {
        pModel = static_pointer_cast<typename IteratorType::NodeType>(pModel)->getModel0();
        ptrList.push_back(pModel);
        ++depth;
    }

    return IteratorType(ptrList, depth);
}


template<typename ModelType,
         typename IteratorType = typename std::conditional<std::is_const<ModelType>::value, ModelConstIterator, ModelIterator>::type,
         typename = typename std::enable_if<std::is_same<Model, typename std::decay<ModelType>::type>::value>::type>
IteratorType pointsEnd(boost::shared_ptr<ModelType> pModel)
{
    return IteratorType(list<shared_ptr<ModelType>>{});
}


template<typename ModelType,
         typename = typename std::enable_if<std::is_same<Model, typename std::decay<ModelType>::type>::value>::type>
double getDistance(shared_ptr<ModelType> pModel0, shared_ptr<ModelType> pModel1, int outputID)
{
    auto const mBegin0 = pointsBegin(pModel0), mEnd0 = pointsEnd(pModel0);
    auto const mBegin1 = pointsBegin(pModel1), mEnd1 = pointsEnd(pModel1);

    auto mIt0 = mBegin0, mIt1 = mBegin1;
    double dist = distanceBetweenDataPoints(*mIt0, *mIt1, outputID);

    for (++mIt1; mIt1 != mEnd1; ++mIt1)
    {
        auto const temp = distanceBetweenDataPoints(*mIt0, *mIt1, outputID);
        if (temp < dist)
        {
            dist = temp;
        }
    }

    for (++mIt0; mIt0 != mEnd0; ++mIt0)
    {
        for (mIt1 = mBegin1; mIt1 != mEnd1; ++mIt1)
        {
            auto const temp = distanceBetweenDataPoints(*mIt0, *mIt1, outputID);
            if (temp < dist)
            {
                dist = temp;
            }
        }
    }

    return dist;
}


double distanceBetweenDataPoints(DataPoint const& point0, DataPoint const& point1, int outputID)
{
    double dist = 0.;

    for (auto it0 = point0.x.begin(), end0 = point0.x.end(), it1 = point1.x.begin();
         it0 != end0; ++it0, ++it1)
    {
        dist += abs(it0->value - it1->value);
    }

    for (auto it0 = point0.t[outputID].begin(), end0 = point0.t[outputID].end(), it1 = point1.t[outputID].begin();
         it0 != end0; ++it0, ++it1)
    {
        dist += abs(it0->value - it1->value);
    }

    return dist;
}




}}
