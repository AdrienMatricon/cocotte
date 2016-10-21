#ifndef COCOTTE_APPROXIMATORS_APPROXIMATOR_H
#define COCOTTE_APPROXIMATORS_APPROXIMATOR_H


#include <vector>
#include <list>
#include <string>
#include <utility>
#include <boost/serialization/vector.hpp>
#include <cocotte/approximators/form.h>
#include <cocotte/models/modeliterator.h>
#include <cocotte/useddimensions.h>



namespace Cocotte {
namespace Approximators {


template <typename ApproximatorType>
class Approximator
{

public:


    // Computes the minimum complexity such that for each number of dimensions
    //   d which allows for forms with a complexity <= maxComplexity
    //   there is at least one with a complexity within [minComplexity, maxComplexity]
    static unsigned int getComplexityRangeLowerBound(
            unsigned int maxNbDimensions,
            unsigned int maxComplexity)
    {
        return ApproximatorType::getComplexityRangeLowerBound_implementation(maxNbDimensions, maxComplexity);
    }

    // Returns all forms of complexity in [minComplexity, maxComplexity]
    // The first part of the result are forms that use no new dimensions
    // The second part of the result are forms that use exactly one new dimension
    // Each are a list of list of forms:
    //   - forms in the same list have the same complexity
    //   - the list of lists is sorted by (non strictly) increasing complexity
    static std::pair<std::list<std::list<Form>>, std::list<std::list<Form>>> getFormsInComplexityRange(
            UsedDimensions const& formerlyUsedDimensions,
            unsigned int minComplexity,
            unsigned int maxComplexity)
    {
        return ApproximatorType::getFormsInComplexityRange_implementation(formerlyUsedDimensions, minComplexity, maxComplexity);
    }

    // Tries to fit the points with a form
    // If success, params are stored within the form
    static bool tryFit(Form& form, unsigned int nbPoints, Models::ModelConstIterator mBegin, Models::ModelConstIterator mEnd, unsigned int outputID, unsigned int dimInOutput)
    {
        return ApproximatorType::tryFit_implementation(form, nbPoints, mBegin, mEnd, outputID, dimInOutput);
    }

    static Form fitOnePoint(double t, unsigned int nbDims)
    {
        return ApproximatorType::fitOnePoint_implementation(t, nbDims);
    }

    // Estimates the value for the given inputs
    static std::vector<double> estimate(Form const& form, std::vector<std::vector<double>> const& x)
    {
        return ApproximatorType::estimate_implementation(form, x);
    }

    // Returns the form as a readable string
    static std::string formToString(Form const& form, std::vector<std::string> inputNames)
    {
        return ApproximatorType::formToString_implementation(form, inputNames);
    }
};



}}



#endif
