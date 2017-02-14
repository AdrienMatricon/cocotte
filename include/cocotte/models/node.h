#ifndef COCOTTE_MODELS_NODE_H
#define COCOTTE_MODELS_NODE_H


#include <vector>
#include <memory>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/vector.hpp>
#include <cocotte/datatypes.h>
#include <cocotte/approximators/form.h>
#include <cocotte/models/modeldistance.h>
#include <cocotte/models/model.h>



namespace Cocotte {
namespace Models {



class Node : public Model
{

private:

    std::vector<Approximators::Form> forms;
    unsigned int nbPoints;
    ModelDistance biggestInnerDistance;
    bool alreadyComputed = false;
    std::shared_ptr<Model> model0;
    std::shared_ptr<Model> model1;
    bool temporary;


public:

    Node() = default;
    Node(std::shared_ptr<Model> model0, std::shared_ptr<Model> model1, bool temporary = false);
    virtual bool isLeaf() const override;
    virtual bool isTemporary() const override;
    virtual unsigned int getNbPoints() const override;
    ModelDistance getBiggestInnerDistance(unsigned int outputID);  // Biggest distance between two merged submodels
    virtual std::vector<Approximators::Form> const& getForms() override;
    void setForms(std::vector<Approximators::Form> const& forms);

    std::shared_ptr<Model> getModel0() const;
    std::shared_ptr<Model> getModel1() const;

    // Serialization
    template<typename Archive>
    friend void serialize(Archive& archive, Node& node, const unsigned int version)
    {
        (void) version; // Unused parameter

        archive & boost::serialization::base_object<Model>(node);
        archive & node.forms;
        archive & node.nbPoints;
        archive & node.model0;
        archive & node.model1;
        archive & node.temporary;
    }

};



}}



#endif
