
#include <algorithm>
using std::max;
#include <vector>
using std::vector;
#include <list>
using std::list;
#include <boost/shared_ptr.hpp>
using boost::shared_ptr;
using boost::static_pointer_cast;
#include <cocotte/approximators/form.h>
using Cocotte::Approximators::Form;
#include <cocotte/models/modeliterator.h>
using Cocotte::Models::getDistance;
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


double Node::getBiggestInnerDistance(int outputID)
{
    if (biggestInnerDistance < 0.)
    {
        // We compute the biggest inner distance for all nodes below this one
        vector<shared_ptr<Model>> children = {model0, model1};
        for (auto& child : children)
        {
            if (child->isLeaf())
            {
                continue;
            }

            auto asNode = static_pointer_cast<Node>(child);
            if (asNode->biggestInnerDistance > 0.)
            {
                continue;
            }


            list<shared_ptr<Node>> predecessors;
            predecessors.push_back(asNode);

            auto current = asNode->model0;

            while (!predecessors.empty())
            {
                // At first we go down until we reach leaves
                // or already computed biggest inner distances
                if (!current->isLeaf())
                {
                    asNode = static_pointer_cast<Node>(current);
                    if (asNode->biggestInnerDistance < 0.)
                    {
                        predecessors.push_back(asNode);
                        current = asNode->model0;
                        continue;
                    }
                }

                // Then we either switch branches or compute
                // biggest inner distances and go back up the tree
                auto child1 = predecessors.back()->model1;
                if (current == child1)
                {
                    // Compute biggest inner distance and go back up the tree
                    asNode = predecessors.back();
                    current = static_pointer_cast<Model>(asNode);
                    predecessors.pop_back();
                    auto child0 = asNode->model0;
                    auto dist = Models::getDistance(child0, child1, outputID);

                    if (!child0->isLeaf())
                    {
                        dist = max(dist, static_pointer_cast<Node>(child0)->biggestInnerDistance);
                    }

                    if (!child1->isLeaf())
                    {
                        dist = max(dist, static_pointer_cast<Node>(child1)->biggestInnerDistance);
                    }

                    asNode->biggestInnerDistance = dist;
                }
                else
                {
                    // Switch branches
                    current = child1;
                }
            }
        }

        // And now we compute it for this node
        biggestInnerDistance = Models::getDistance(model0, model1, outputID);

        if (model0->isLeaf())
        {
            biggestInnerDistance = max(biggestInnerDistance, static_pointer_cast<Node>(model0)->biggestInnerDistance);
        }

        if (model1->isLeaf())
        {
            biggestInnerDistance = max(biggestInnerDistance, static_pointer_cast<Node>(model1)->biggestInnerDistance);
        }
    }

    return biggestInnerDistance;
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


