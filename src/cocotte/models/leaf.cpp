
#include <vector>
using std::vector;
#include <memory>
using std::shared_ptr;
#include <cocotte/approximators/form.h>
using Cocotte::Approximators::Form;
#include <cocotte/models/leaf.h>

namespace Cocotte {
namespace Models {



Leaf::Leaf(vector<Form> const& f, shared_ptr<DataPoint const> p, bool temp) : forms(f), pointAddress(p), temporary(temp)
{}


bool Leaf::isLeaf() const
{
    return true;
}


bool Leaf::isTemporary() const
{
    return temporary;
}


unsigned int Leaf::getNbPoints() const
{
    return 1;
}


vector<Form> const& Leaf::getForms()
{
    return forms;
}


DataPoint const& Leaf::getPoint()
{
    return *pointAddress;
}


DataPoint const& Leaf::getPoint() const
{
    return *pointAddress;
}


shared_ptr<DataPoint const> Leaf::getPointAddress() const
{
    return pointAddress;
}


}}


