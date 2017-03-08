#ifndef COCOTTE_MODELS_LEAF_H
#define COCOTTE_MODELS_LEAF_H


#include <vector>
#include <memory>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <cocotte/datatypes.h>
#include <cocotte/models/model.h>



namespace Cocotte {
namespace Models {



template<typename ApproximatorType>
class Leaf : public Model<ApproximatorType>
{

private:

    std::vector<Approximators::Form<ApproximatorType>> forms;
    std::shared_ptr<DataPoint const> pointAddress;
    bool temporary;


public:

    Leaf() = default;
    Leaf(std::vector<Approximators::Form<ApproximatorType>> const& forms,
         std::shared_ptr<DataPoint const> pointAddress, bool temporary = false);
    virtual bool isLeaf() const override;
    virtual bool isTemporary() const override;
    virtual unsigned int getNbPoints() const override;
    virtual std::vector<Approximators::Form<ApproximatorType>> const& getForms() override;

    DataPoint const& getPoint();
    DataPoint const& getPoint() const;
    std::shared_ptr<DataPoint const> getPointAddress() const;

    // Serialization
    template<typename Archive>
    friend void serialize(Archive& archive, Leaf& leaf, const unsigned int version)
    {
        (void) version; // Unused parameter

        archive & boost::serialization::base_object<Model<ApproximatorType>>(leaf);
        archive & leaf.forms;
        archive & leaf.pointAddress;
        archive & leaf.temporary;
    }

};



}}

#include <cocotte/models/leaf.hpp>



#endif
