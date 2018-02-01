
#include <vector>
#include <list>
#include <memory>
#include <cocotte/models/modeliterator.h>
#include <cocotte/models/node.h>

namespace Cocotte {
namespace Models {



template<typename ApproximatorType>
Node<ApproximatorType>::Node(std::vector<std::shared_ptr<Model<ApproximatorType>>> mod,
                             bool temp)
    : models(mod), temporary(temp)
{
    nbPoints = 0;
    for (auto const& m : mod)
    {
        nbPoints += m->getNbPoints();
    }
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
        // An element of the outer list corresponds to one depth of the tree
        // The inner list contains the unexplored elements of a given depth
        // (this is a depth-first exploration of the tree)
        list<vector<shared_ptr<ModelType>>> toProcess{models};

        while (!toProcess.empty())
        {
            auto current = toProcess.back().back();

            if (!current->isLeaf())
            {
                auto const asNode = static_pointer_cast<NodeType>(current);
                if (!asNode->alreadyComputed)
                {
                    toProcess.push_back(asNode->getModels());
                    continue;
                }
            }

            toProcess.back().pop_back();

            while (toProcess.back().empty())
            {
                // We just finished exploring a node
                toProcess.pop_back();
                if (toProcess.empty())
                {
                    break;
                }

                auto node = static_pointer_cast<NodeType>(toProcess.back().back());

                // We compute the distance between the submodels of the node
                {
                    auto const mod = node->getModels();

                    auto mIt0 = mod.begin();
                    auto mIt1 = mod.begin(); ++mIt1;    // We assume there are at least 2 submodels
                    auto const mEnd = mod.end();

                    auto dist = Models::getDistance<ApproximatorType>(*mIt0, *mIt1, outputID);
                    ++mIt1;

                    do
                    {
                        if (!(*mIt0)->isLeaf())
                        {
                            auto const temp = static_pointer_cast<NodeType>(*mIt0)->biggestInnerDistance;
                            if (temp > dist)
                            {
                                dist = temp;
                            }
                        }

                        for (; mIt1 != mEnd; ++mIt1)
                        {
                            auto const temp = Models::getDistance<ApproximatorType>(*mIt0, *mIt1, outputID);
                            if (temp > dist)
                            {
                                dist = temp;
                            }
                        }

                        ++mIt0; mIt1 = mIt0; ++mIt1;
                    } while (mIt0 != mEnd);

                    node->biggestInnerDistance = dist;
                    node->alreadyComputed = true;
                }

                toProcess.back().pop_back();
            }
        }

        // Now we compute the biggest inner distance overall
        {
            auto mIt0 = models.begin();
            auto mIt1 = models.begin(); ++mIt1;    // We assume there are at least 2 submodels
            auto const mEnd = models.end();

            auto dist = Models::getDistance<ApproximatorType>(*mIt0, *mIt1, outputID);
            ++mIt1;

            do
            {
                if (!(*mIt0)->isLeaf())
                {
                    auto const temp = static_pointer_cast<NodeType>(*mIt0)->biggestInnerDistance;
                    if (temp > dist)
                    {
                        dist = temp;
                    }
                }

                for (; mIt1 != mEnd; ++mIt1)
                {
                    auto const temp = Models::getDistance<ApproximatorType>(*mIt0, *mIt1, outputID);
                    if (temp > dist)
                    {
                        dist = temp;
                    }
                }

                ++mIt0; mIt1 = mIt0; ++mIt1;
            } while (mIt0 != mEnd);

            biggestInnerDistance = dist;
            alreadyComputed = true;
        }
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
std::vector<std::shared_ptr<Model<ApproximatorType>>> const& Node<ApproximatorType>::getModels() const
{
    return models;
}

template<typename ApproximatorType>
std::shared_ptr<Model<ApproximatorType>> Node<ApproximatorType>::getModel(unsigned int i) const
{
    return models[i];
}



}}


