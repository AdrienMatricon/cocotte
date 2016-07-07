
#include <vector>
using std::vector;
#include <boost/shared_ptr.hpp>
using boost::shared_ptr;
#include <cocotte/approximators/form.h>
using Cocotte::Approximators::Form;
#include <cocotte/models/node.h>

namespace Cocotte {
namespace Models {



Node::Node(shared_ptr<Model> m0, shared_ptr<Model> m1, bool temp) : model0(m0), model1(m1), temporary(temp)
{
    nbPoints = model0->getNbPoints() + model1->getNbPoints();
}


bool Node::isLeaf() const
{
    return false;
}


bool Node::isTemporary() const
{
    return temporary;
}


int Node::getNbPoints() const
{
    return nbPoints;
}


vector<Form> const& Node::getForms()
{
    return forms;
}


void Node::setForms(vector<Form> const& f)
{
    forms = f;
}


shared_ptr<Model> Node::getModel0() const
{
    return model0;
}

shared_ptr<Model> Node::getModel1() const
{
    return model1;
}



}}


