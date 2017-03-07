#ifndef COCOTTE_APPROXIMATORS_APPROXIMATOR_H
#define COCOTTE_APPROXIMATORS_APPROXIMATOR_H


#include <vector>
#include <list>
#include <string>
#include <tuple>
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
    //   d in [1, maxNbDimensions] which allows for forms with a complexity <= maxComplexity
    //   there is at least one with a complexity within [minComplexity, maxComplexity]
    static unsigned int getComplexityRangeLowerBound(
            unsigned int maxNbDimensions,
            unsigned int maxComplexity)
    {
        return ApproximatorType::getComplexityRangeLowerBound_implementation(maxNbDimensions, maxComplexity);
    }

    // Returns all forms of complexity in [minComplexity, maxComplexity],
    // using exactly nbNewDimensions dimensions not in formerlyUsedDimensions
    // Those forms are returned as a lists of sublists of forms, such that:
    //   - forms in the same sublist have the same complexity
    //   - sublists are sorted by (non strictly) increasing complexity
    static std::list<std::list<Form>> getFormsInComplexityRange(
            UsedDimensions const& formerlyUsedDimensions,
            unsigned int nbNewDimensions,
            unsigned int minComplexity,
            unsigned int maxComplexity)
    {
        return ApproximatorType::getFormsInComplexityRange_implementation(
                    formerlyUsedDimensions, nbNewDimensions, minComplexity, maxComplexity);
    }

    // Tries to fit the points with a form
    // If it succeeds, params are stored within the form
    // The function returns whether is was a success
    //   and (only if it succeeded) the fitness of the form that was found
    //   (the lower the better)
    static std::tuple<bool,double> tryFit(Form& form, unsigned int nbPoints,
                                          Models::ModelConstIterator mBegin, Models::ModelConstIterator mEnd,
                                          unsigned int outputID, unsigned int dimInOutput)
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
