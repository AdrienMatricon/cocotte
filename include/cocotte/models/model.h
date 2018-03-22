#ifndef COCOTTE_MODELS_MODEL_H
#define COCOTTE_MODELS_MODEL_H


#include <vector>
#include <cocotte/approximators/forward.h>
#include <boost/serialization/assume_abstract.hpp>
#include <boost/serialization/vector.hpp>

namespace Cocotte {
namespace Models {



template<typename ApproximatorType>
class Model
{

public:

    virtual bool isLeaf() const = 0;
    virtual unsigned int getNbPoints() const = 0;
    virtual unsigned int getSumOfComplexities(unsigned int nbOutputDims) const = 0;
    virtual std::vector<Approximators::Form<ApproximatorType>> const& getForms() = 0;
    virtual ~Model(){}

    // Serialization
    template<typename Archive>
    friend void serialize(Archive& archive, Model& model, const unsigned int version)
    {
        // Unused parameters
        (void) archive;
        (void) model;
        (void) version;
    }

};

BOOST_SERIALIZATION_ASSUME_ABSTRACT(Model)



}}



#endif
