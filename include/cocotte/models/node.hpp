
#include <vector>
#include <list>
#include <memory>
#include <cocotte/models/modeliterator.h>
#include <cocotte/models/node.h>

namespace Cocotte {
namespace Models {



template<typename ApproximatorType>
Node<ApproximatorType>::Node(std::shared_ptr<Model<ApproximatorType>> m0,
                             std::shared_ptr<Model<ApproximatorType>> m1,
                             bool temp)
    : model0(m0), model1(m1), temporary(temp)
{
    nbPoints = model0->getNbPoints() + model1->getNbPoints();
}


template<typename ApproximatorType>
bool Node<ApproximatorType>::isLeaf() const
{
    return false;
}


template<typename ApproximatorType>
bool Node<ApproximatorType>::isTemporary() const
{
    return temporary;
}


template<typename ApproximatorType>
unsigned int Node<ApproximatorType>::getNbPoints() const
{
    return nbPoints;
}


template<typename ApproximatorType>
ModelDistance Node<ApproximatorType>::getBiggestInnerDistance(unsigned int outputID)
{
    using std::vector;
    using std::list;
    using std::shared_ptr;
    using std::static_pointer_cast;
    using ModelType = Model<ApproximatorType>;
    using NodeType = Node<ApproximatorType>;

    if (!alreadyComputed)
    {
        // We compute the biggest inner distance for all nodes below this one
        vector<shared_ptr<ModelType>> children = {model0, model1};
        for (auto& child : children)
        {
            if (child->isLeaf())
            {
                continue;
            }

            auto asNode = static_pointer_cast<NodeType>(child);
            if (asNode->alreadyComputed)
            {
                continue;
            }


            list<shared_ptr<NodeType>> predecessors;
            predecessors.push_back(asNode);

            auto current = asNode->model0;

            while (!predecessors.empty())
            {
                // At first we go down until we reach leaves
                // or already computed biggest inner distances
                if (!current->isLeaf())
                {
                    asNode = static_pointer_cast<NodeType>(current);
                    if (!asNode->alreadyComputed)
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
                    current = static_pointer_cast<ModelType>(asNode);
                    predecessors.pop_back();
                    auto child0 = asNode->model0;
                    auto dist = Models::getDistance<ApproximatorType>(child0, child1, outputID);

                    if (!child0->isLeaf() && (dist < static_pointer_cast<NodeType>(child0)->biggestInnerDistance))
                    {
                        dist = static_pointer_cast<NodeType>(child0)->biggestInnerDistance;
                    }

                    if (!child1->isLeaf() && (dist < static_pointer_cast<NodeType>(child1)->biggestInnerDistance))
                    {
                        dist = static_pointer_cast<NodeType>(child1)->biggestInnerDistance;
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
        biggestInnerDistance = Models::getDistance<ApproximatorType>(model0, model1, outputID);

        if (model0->isLeaf() && (biggestInnerDistance < static_pointer_cast<NodeType>(model0)->biggestInnerDistance))
        {
            biggestInnerDistance = static_pointer_cast<NodeType>(model0)->biggestInnerDistance;
        }

        if (model1->isLeaf() && (biggestInnerDistance < static_pointer_cast<NodeType>(model1)->biggestInnerDistance))
        {
            biggestInnerDistance = static_pointer_cast<NodeType>(model1)->biggestInnerDistance;
        }

        alreadyComputed = true;
    }

    return biggestInnerDistance;
}


template<typename ApproximatorType>
std::vector<Approximators::Form<ApproximatorType>> const& Node<ApproximatorType>::getForms()
{
    return forms;
}


template<typename ApproximatorType>
void Node<ApproximatorType>::setForms(std::vector<Approximators::Form<ApproximatorType>> const& f)
{
    forms = f;
}


template<typename ApproximatorType>
std::shared_ptr<Model<ApproximatorType>> Node<ApproximatorType>::getModel0() const
{
    return model0;
}

template<typename ApproximatorType>
std::shared_ptr<Model<ApproximatorType>> Node<ApproximatorType>::getModel1() const
{
    return model1;
}



}}


