#ifndef COCOTTE_MODELS_MODEL_H
#define COCOTTE_MODELS_MODEL_H


#include <vector>
#include <boost/serialization/assume_abstract.hpp>
#include <boost/serialization/vector.hpp>
#include <cocotte/approximators/form.h>

namespace Cocotte {
namespace Models {



class Model
{

public:

    virtual bool isLeaf() const = 0;
    virtual bool isTemporary() const = 0;
    virtual int getNbPoints() const = 0;
    virtual std::vector<Approximators::Form> const& getForms() = 0;
    virtual ~Model(){}

    // Serialization
    template<typename Archive>
    friend void serialize(Archive& archive, Model& model, const unsigned int version){}

};

BOOST_SERIALIZATION_ASSUME_ABSTRACT(Model)



}}



#endif
