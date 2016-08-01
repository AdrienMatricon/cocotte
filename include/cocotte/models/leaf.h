#ifndef COCOTTE_MODELS_LEAF_H
#define COCOTTE_MODELS_LEAF_H


#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <cocotte/datatypes.h>
#include <cocotte/approximators/form.h>
#include <cocotte/models/model.h>



namespace Cocotte {
namespace Models {



class Leaf : public Model
{

private:

    std::vector<Approximators::Form> forms;
    boost::shared_ptr<DataPoint const> pointAddress;
    bool temporary;


public:

    Leaf() = default;
    Leaf(std::vector<Approximators::Form> const& forms, boost::shared_ptr<DataPoint const> pointAddress, bool temporary = false);
    virtual bool isLeaf() const override;
    virtual bool isTemporary() const override;
    virtual size_t getNbPoints() const override;
    virtual std::vector<Approximators::Form> const& getForms() override;

    DataPoint const& getPoint();
    DataPoint const& getPoint() const;
    boost::shared_ptr<DataPoint const> getPointAddress() const;

    // Serialization
    template<typename Archive>
    friend void serialize(Archive& archive, Leaf& leaf, const unsigned int version)
    {
        archive & boost::serialization::base_object<Model>(leaf);
        archive & leaf.forms;
        archive & leaf.pointAddress;
        archive & leaf.temporary;
    }

};



}}



#endif
