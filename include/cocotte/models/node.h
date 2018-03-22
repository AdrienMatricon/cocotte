#ifndef COCOTTE_MODELS_NODE_H
#define COCOTTE_MODELS_NODE_H


#include <vector>
#include <memory>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/vector.hpp>
#include <cocotte/datatypes.h>
#include <cocotte/models/modeldistance.h>
#include <cocotte/models/model.h>



namespace Cocotte {
namespace Models {



template<typename ApproximatorType>
class Node : public Model<ApproximatorType>
{

private:

    std::vector<Approximators::Form<ApproximatorType>> forms;
    unsigned int nbPoints;
    ModelDistance biggestInnerDistance;
    bool alreadyComputed = false;
    std::vector<std::shared_ptr<Model<ApproximatorType>>> submodels;


public:

    Node() = default;
    Node(std::vector<std::shared_ptr<Model<ApproximatorType>>> submodels);
    virtual bool isLeaf() const override;
    virtual unsigned int getNbPoints() const override;
    virtual unsigned int getSumOfComplexities(unsigned int nbOutputDims) const override;
    ModelDistance getBiggestInnerDistance(unsigned int outputID);  // Biggest distance between merged submodels
    virtual std::vector<Approximators::Form<ApproximatorType>> const& getForms() override;
    void setForms(std::vector<Approximators::Form<ApproximatorType>> const& forms);

    std::vector<std::shared_ptr<Model<ApproximatorType>>> const& getSubmodels() const;
    std::shared_ptr<Model<ApproximatorType>> getSubmodel(unsigned int i) const;

    // Serialization
    template<typename Archive>
    friend void serialize(Archive& archive, Node& node, const unsigned int version)
    {
        (void) version; // Unused parameter

        archive & boost::serialization::base_object<Model<ApproximatorType>>(node);
        archive & node.forms;
        archive & node.nbPoints;
        archive & node.submodels;
    }

};



}}

#include <cocotte/models/node.hpp>



#endif
