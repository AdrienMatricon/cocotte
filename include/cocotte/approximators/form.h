#ifndef COCOTTE_APPROXIMATORS_FORM_H
#define COCOTTE_APPROXIMATORS_FORM_H


#include <vector>
#include <boost/serialization/vector.hpp>
#include <cocotte/useddimensions.h>



namespace Cocotte {
namespace Approximators {


struct Form
{
    UsedDimensions usedDimensions;
    std::vector<double> params;
    unsigned int complexity;
    unsigned int other;

    Form() = default;
    Form(UsedDimensions uD) : usedDimensions(uD) {}

    // Serialization
    template<typename Archive>
    friend void serialize(Archive& archive, Form& form, const unsigned int version)
    {
        archive & form.usedDimensions;
        archive & form.params;
        archive & form.complexity;
        archive & form.other;
    }
};



}}



#endif
