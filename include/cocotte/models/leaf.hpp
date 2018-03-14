
#include <vector>
#include <memory>
#include <cocotte/models/leaf.h>

namespace Cocotte {
namespace Models {



template<typename ApproximatorType>
Leaf<ApproximatorType>::Leaf(std::vector<Approximators::Form<ApproximatorType>> const& f,
           std::shared_ptr<DataPoint const> p)
    : forms(f), pointAddress(p)
{}


template<typename ApproximatorType>
bool Leaf<ApproximatorType>::isLeaf() const
{
    return true;
}


template<typename ApproximatorType>
unsigned int Leaf<ApproximatorType>::getNbPoints() const
{
    return 1;
}


template<typename ApproximatorType>
std::vector<Approximators::Form<ApproximatorType>> const& Leaf<ApproximatorType>::getForms()
{
    return forms;
}


template<typename ApproximatorType>
DataPoint const& Leaf<ApproximatorType>::getPoint()
{
    return *pointAddress;
}


template<typename ApproximatorType>
DataPoint const& Leaf<ApproximatorType>::getPoint() const
{
    return *pointAddress;
}


template<typename ApproximatorType>
std::shared_ptr<DataPoint const> Leaf<ApproximatorType>::getPointAddress() const
{
    return pointAddress;
}


}}


