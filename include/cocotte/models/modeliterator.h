#ifndef COCOTTE_MODELS_MODELITERATOR_H
#define COCOTTE_MODELS_MODELITERATOR_H


#include <iterator>
#include <type_traits>
#include <list>
#include <vector>
#include <utility>
#include <boost/shared_ptr.hpp>
#include <type_traits>
#include <cocotte/datatypes.h>
#include <cocotte/models/leaf.h>
#include <cocotte/models/node.h>



namespace Cocotte {
namespace Models {



template <typename ModelType, typename PointType>
class ModelIt : public std::iterator<std::input_iterator_tag, PointType>
{

private:

    std::list<boost::shared_ptr<ModelType>> treeBranch;
    int depth = 0;


public:

    using NodeType = typename std::conditional<std::is_const<ModelType>::value, Node const, Node>::type;
    using LeafType = typename std::conditional<std::is_const<ModelType>::value, Leaf const, Leaf>::type;

    // Constructors
    ModelIt(std::list<boost::shared_ptr<ModelType>> treeBranch, int depth=0);
    ModelIt(ModelIt const& mIt);

    // Accessors
    std::list<boost::shared_ptr<Model>> const& getTreeBranch() const;
    int getDepth() const;

    // Operators
    boost::shared_ptr<PointType const> getSharedPointer();
    PointType const& operator*();
    PointType const* operator->();

    ModelIt& operator++();
    ModelIt operator++(int);

    bool operator==(ModelIt const& rhs);
    bool operator!=(const ModelIt& rhs);

};



using ModelIterator = ModelIt<Model, DataPoint>;
using ModelConstIterator = ModelIt<Model const, DataPoint const>;


template<typename ModelType,
         typename IteratorType = typename std::conditional<std::is_const<ModelType>::value, ModelConstIterator, ModelIterator>::type,
         typename = typename std::enable_if<std::is_same<Model, typename std::decay<ModelType>::type>::value>::type>
IteratorType pointsBegin(boost::shared_ptr<ModelType> pModel);

template<typename ModelType,
         typename IteratorType = typename std::conditional<std::is_const<ModelType>::value, ModelConstIterator, ModelIterator>::type,
         typename = typename std::enable_if<std::is_same<Model, typename std::decay<ModelType>::type>::value>::type>
IteratorType pointsEnd(boost::shared_ptr<ModelType> pModel);

template<typename ModelType,
         typename = typename std::enable_if<std::is_same<Model, typename std::decay<ModelType>::type>::value>::type>
double getDistance(boost::shared_ptr<ModelType> pModel0, boost::shared_ptr<ModelType> pModel1, int outputID);

inline double distanceBetweenDataPoints(DataPoint const& point0, DataPoint const& point1, int outputID);



}}


#include <cocotte/models/modeliterator.hpp>



#endif
