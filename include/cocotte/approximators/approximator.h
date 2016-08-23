#ifndef COCOTTE_APPROXIMATORS_APPROXIMATOR_H
#define COCOTTE_APPROXIMATORS_APPROXIMATOR_H


#include <vector>
#include <list>
#include <string>
#include <boost/serialization/vector.hpp>
#include <cocotte/approximators/form.h>
#include <cocotte/models/modeliterator.h>
#include <cocotte/useddimensions.h>



namespace Cocotte {
namespace Approximators {


class Approximator
{

public:


    // Returns the most complex forms under a certain complexity,
    // for each combination of the formerly used dimensions and at most one other.
    virtual std::list<std::list<Form>> getMostComplexForms(
            UsedDimensions const& formerlyUsedDimensions,
            unsigned int maxComplexity) = 0;

    // Tries to fit the points with a form
    // If success, params are stored within the form
    virtual bool tryFit(Form& form, unsigned int nbPoints, Models::ModelConstIterator mBegin, Models::ModelConstIterator mEnd, unsigned int outputID, unsigned int dimInOutput) = 0;

    virtual Form fitOnePoint(double t, unsigned int nbDims) = 0;

    // Estimates the value for the given inputs
    virtual std::vector<double> estimate(Form const& form, std::vector<std::vector<double>> const& x) = 0;

    // Returns the form as a readable string
    virtual std::string formToString(Form const& form, std::vector<std::string> inputNames) = 0;

};



}}



#endif
