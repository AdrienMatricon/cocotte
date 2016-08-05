#ifndef COCOTTE_MODELS_NODE_H
#define COCOTTE_MODELS_NODE_H


#include <vector>
#include <memory>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/vector.hpp>
#include <cocotte/datatypes.h>
#include <cocotte/approximators/form.h>
#include <cocotte/models/model.h>



namespace Cocotte {
namespace Models {



class Node : public Model
{

private:

    std::vector<Approximators::Form> forms;
    int nbPoints;
    double biggestInnerDistance = -1.;
    std::shared_ptr<Model> model0;
    std::shared_ptr<Model> model1;
    bool temporary;


public:

    Node() = default;
    Node(std::shared_ptr<Model> model0, std::shared_ptr<Model> model1, bool temporary = false);
    virtual bool isLeaf() const override;
    virtual bool isTemporary() const override;
    virtual size_t getNbPoints() const override;
    double getBiggestInnerDistance(int outputID);   // Biggest distance between two merged submodels
    virtual std::vector<Approximators::Form> const& getForms() override;
    void setForms(std::vector<Approximators::Form> const& forms);

    std::shared_ptr<Model> getModel0() const;
    std::shared_ptr<Model> getModel1() const;

    // Serialization
    template<typename Archive>
    friend void serialize(Archive& archive, Node& node, const unsigned int version)
    {
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
