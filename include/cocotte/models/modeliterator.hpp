
#include <cmath>
#include <list>
#include <utility>
#include <memory>
#include <cocotte/distance.h>
#include <cocotte/models/leaf.h>
#include <cocotte/models/node.h>
#include <cocotte/models/modeliterator.h>


namespace Cocotte {
namespace Models {



// Constructors
template <typename ApproximatorType, typename ModelType, typename PointType>
ModelIt<ApproximatorType, ModelType, PointType>::ModelIt(
        std::list<std::shared_ptr<ModelType>> t, unsigned int d) : treeBranch(t), depth(d)
{}


template <typename ApproximatorType, typename ModelType, typename PointType>
ModelIt<ApproximatorType, ModelType, PointType>::ModelIt(
        ModelIt const& mIt) : treeBranch(mIt.treeBranch), depth(mIt.depth)
{}



// Accessors
template <typename ApproximatorType, typename ModelType, typename PointType>
std::list<std::shared_ptr<Model<ApproximatorType>>> const& ModelIt<ApproximatorType, ModelType, PointType>::getTreeBranch() const
{
    return treeBranch;
}


template <typename ApproximatorType, typename ModelType, typename PointType>
unsigned int ModelIt<ApproximatorType, ModelType, PointType>::getDepth() const
{
    return depth;
}



// Operators
template <typename ApproximatorType, typename ModelType, typename PointType>
std::shared_ptr<PointType const> ModelIt<ApproximatorType, ModelType, PointType>::getSharedPointer()
{
    using std::shared_ptr;
    using std::static_pointer_cast;

    shared_ptr<LeafType> const leaf = static_pointer_cast<LeafType>(treeBranch.back());
    return leaf->getPointAddress();
}


template <typename ApproximatorType, typename ModelType, typename PointType>
PointType const& ModelIt<ApproximatorType, ModelType, PointType>::operator*()
{
    using std::shared_ptr;
    using std::static_pointer_cast;

    shared_ptr<LeafType> const leaf = static_pointer_cast<LeafType>(treeBranch.back());
    return leaf->getPoint();
}


template <typename ApproximatorType, typename ModelType, typename PointType>
PointType const* ModelIt<ApproximatorType, ModelType, PointType>::operator->()
{
    using std::shared_ptr;
    using std::static_pointer_cast;

    shared_ptr<LeafType> const leaf = static_pointer_cast<LeafType>(treeBranch.back());
    return &(leaf->getPoint());
}


template <typename ApproximatorType, typename ModelType, typename PointType>
ModelIt<ApproximatorType, ModelType, PointType>& ModelIt<ApproximatorType, ModelType, PointType>::operator++()
{
    using std::shared_ptr;
    using std::static_pointer_cast;

    ModelType* lastVisited = treeBranch.back().get();
    treeBranch.pop_back();
    --depth;

    if (treeBranch.empty())
    {
        return *this;
    }

    shared_ptr<NodeType> currentNode = static_pointer_cast<NodeType>(treeBranch.back());

    // We go up the tree branch until we don't come from the last child
    while (currentNode->getModels().back().get() == lastVisited)
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

    // We get the next child
    shared_ptr<ModelType> nextChild;
    {
        auto cIt = currentNode->getModels().begin();
        while(cIt->get() != lastVisited)
        {
            ++cIt;
        }
        ++cIt;
        nextChild = *cIt;
    }

    treeBranch.push_back(nextChild);
    ++depth;

    while (!nextChild->isLeaf())
    {
        nextChild = static_pointer_cast<NodeType>(nextChild)->getModel(0);
        treeBranch.push_back(nextChild);
        ++depth;
    }

    return *this;
}


template <typename ApproximatorType, typename ModelType, typename PointType>
ModelIt<ApproximatorType, ModelType, PointType> ModelIt<ApproximatorType, ModelType, PointType>::operator++(int)
{
    ModelIt<ApproximatorType, ModelType, PointType> tmp(*this);
    operator++();
    return tmp;
}


template <typename ApproximatorType, typename ModelType, typename PointType>
bool ModelIt<ApproximatorType, ModelType, PointType>::operator==(
        ModelIt<ApproximatorType, ModelType, PointType> const& rhs)
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


template <typename ApproximatorType, typename ModelType, typename PointType>
bool ModelIt<ApproximatorType, ModelType, PointType>::operator!=(
        const ModelIt<ApproximatorType, ModelType, PointType>& rhs)
{
    return !operator==(rhs);
}


// Iterators
template<typename ApproximatorType, typename ModelType,
         typename IteratorType = typename std::conditional<std::is_const<ModelType>::value, ModelConstIterator<ApproximatorType>, ModelIterator<ApproximatorType>>::type,
         typename = typename std::enable_if<std::is_same<Model<ApproximatorType>, typename std::decay<ModelType>::type>::value>::type>
IteratorType pointsBegin(std::shared_ptr<ModelType> pModel)
{
    using std::list;
    using std::shared_ptr;
    using std::static_pointer_cast;

    list<shared_ptr<ModelType>> ptrList(1, pModel);
    unsigned int depth = 1;

    while (!pModel->isLeaf())
    {
        pModel = static_pointer_cast<typename IteratorType::NodeType>(pModel)->getModel(0);
        ptrList.push_back(pModel);
        ++depth;
    }

    return IteratorType(ptrList, depth);
}


template<typename ApproximatorType, typename ModelType,
         typename IteratorType = typename std::conditional<std::is_const<ModelType>::value, ModelConstIterator<ApproximatorType>, ModelIterator<ApproximatorType>>::type,
         typename = typename std::enable_if<std::is_same<Model<ApproximatorType>, typename std::decay<ModelType>::type>::value>::type>
IteratorType pointsEnd(std::shared_ptr<ModelType> pModel)
{
    using std::list;
    using std::shared_ptr;

    (void) pModel;  // Unused parameter

    return IteratorType(list<shared_ptr<ModelType>>{});
}


template<typename ApproximatorType, typename ModelType,
         typename = typename std::enable_if<std::is_same<Model<ApproximatorType>, typename std::decay<ModelType>::type>::value>::type>
ModelDistance getDistance(std::shared_ptr<ModelType> pModel0, std::shared_ptr<ModelType> pModel1, unsigned int outputID)
{
    auto const mBegin0 = pointsBegin<ApproximatorType>(pModel0), mEnd0 = pointsEnd<ApproximatorType>(pModel0);
    auto const mBegin1 = pointsBegin<ApproximatorType>(pModel1), mEnd1 = pointsEnd<ApproximatorType>(pModel1);

    auto mIt0 = mBegin0, mIt1 = mBegin1;
    ModelDistance dist = ModelDistance(*mIt0, *mIt1, outputID);

    for (++mIt1; mIt1 != mEnd1; ++mIt1)
    {
        auto const temp = ModelDistance(*mIt0, *mIt1, outputID);
        if (temp < dist)
        {
            dist = temp;
        }
    }

    for (++mIt0; mIt0 != mEnd0; ++mIt0)
    {
        for (mIt1 = mBegin1; mIt1 != mEnd1; ++mIt1)
        {
            auto const temp = ModelDistance(*mIt0, *mIt1, outputID);
            if (temp < dist)
            {
                dist = temp;
            }
        }
    }

    return dist;
}




}}
