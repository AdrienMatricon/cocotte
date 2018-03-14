#ifndef COCOTTE_MODELS_MODELITERATOR_H
#define COCOTTE_MODELS_MODELITERATOR_H


#include <iterator>
#include <type_traits>
#include <list>
#include <vector>
#include <utility>
#include <memory>
#include <type_traits>
#include <cocotte/datatypes.h>
#include <cocotte/models/modeldistance.h>



namespace Cocotte {
namespace Models {



// Forward declarations
template<typename ApproximatorType> class Model;
template<typename ApproximatorType> class Leaf;
template<typename ApproximatorType> class Node;



template <typename ApproximatorType, typename ModelType, typename PointType>
class ModelIt : public std::iterator<std::input_iterator_tag, PointType>
{

private:

    std::list<std::shared_ptr<ModelType>> treeBranch;
    unsigned int depth = 0;


public:

    using NodeType = typename std::conditional<std::is_const<ModelType>::value, Node<ApproximatorType> const, Node<ApproximatorType>>::type;
    using LeafType = typename std::conditional<std::is_const<ModelType>::value, Leaf<ApproximatorType> const, Leaf<ApproximatorType>>::type;

    // Constructors
    ModelIt(std::list<std::shared_ptr<ModelType>> treeBranch, unsigned int depth=0);
    ModelIt(ModelIt const& mIt);

    // Accessors
    std::list<std::shared_ptr<Model<ApproximatorType>>> const& getTreeBranch() const;
    unsigned int getDepth() const;

    // Operators
    std::shared_ptr<PointType const> getSharedPointer();
    PointType const& operator*();
    PointType const* operator->();

    ModelIt& operator++();
    ModelIt operator++(int);

    bool operator==(ModelIt const& rhs);
    bool operator!=(const ModelIt& rhs);

};



template<typename ApproximatorType>
using ModelIterator = ModelIt<ApproximatorType, Model<ApproximatorType>, DataPoint>;
template<typename ApproximatorType>
using ModelConstIterator = ModelIt<ApproximatorType, Model<ApproximatorType> const, DataPoint const>;


template<typename ApproximatorType, typename ModelType,
         typename IteratorType = typename std::conditional<std::is_const<ModelType>::value, ModelConstIterator<ApproximatorType>, ModelIterator<ApproximatorType>>::type,
         typename = typename std::enable_if<std::is_same<Model<ApproximatorType>, typename std::decay<ModelType>::type>::value>::type>
IteratorType pointsBegin(std::shared_ptr<ModelType> pModel);

template<typename ApproximatorType, typename ModelType,
         typename IteratorType = typename std::conditional<std::is_const<ModelType>::value, ModelConstIterator<ApproximatorType>, ModelIterator<ApproximatorType>>::type,
         typename = typename std::enable_if<std::is_same<Model<ApproximatorType>, typename std::decay<ModelType>::type>::value>::type>
IteratorType pointsEnd(std::shared_ptr<ModelType> pModel);

template<typename ApproximatorType, typename ModelType,
         typename = typename std::enable_if<std::is_same<Model<ApproximatorType>, typename std::decay<ModelType>::type>::value>::type>
ModelDistance getDistance(std::shared_ptr<ModelType> pModel0, std::shared_ptr<ModelType> pModel1, unsigned int outputID);



}}


#include <cocotte/models/modeliterator.hpp>



#endif
