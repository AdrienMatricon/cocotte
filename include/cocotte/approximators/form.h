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
    unsigned int complexity = 0;
    unsigned int other = 0;

    Form() = default;
    Form(UsedDimensions uD) : usedDimensions(uD) {}

    // Serialization
    template<typename Archive>
    friend void serialize(Archive& archive, Form& form, const unsigned int version)
    {
        (void) version; // Unused parameter

        archive & form.usedDimensions;
        archive & form.params;
        archive & form.complexity;
        archive & form.other;
    }

    // Display
    friend std::ostream& operator<< (std::ostream& out, Form const& form)
    {
        out << "Complexity " << form.complexity
            << " (other: " << form.other << ")"
            << " of dimensions " << form.usedDimensions;

        return out;
    }
};



}}



#endif
