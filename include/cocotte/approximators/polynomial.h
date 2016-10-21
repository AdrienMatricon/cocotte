#ifndef COCOTTE_APPROXIMATORS_POLYNOMIAL_H
#define COCOTTE_APPROXIMATORS_POLYNOMIAL_H


#include <cocotte/approximators/approximator.h>



namespace Cocotte {
namespace Approximators {



class Polynomial : public Approximator<Polynomial>
{
    friend Approximator<Polynomial>;


private:

    // Computes the minimum complexity such that for each number of dimensions
    //   d which allows for forms with a complexity <= maxComplexity
    //   there is at least one with a complexity within [minComplexity, maxComplexity]
    static unsigned int getComplexityRangeLowerBound_implementation(
            unsigned int maxNbDimensions,
            unsigned int maxComplexity);

    // Returns all forms of complexity in [minComplexity, maxComplexity]
    // The first part of the result are forms that use no new dimensions
    // The second part of the result are forms that use exactly one new dimension
    // Each are a list of list of forms:
    //   - forms in the same list have the same complexity
    //   - the list of lists is sorted by (non strictly) increasing complexity
    static std::pair<std::list<std::list<Form>>, std::list<std::list<Form>>> getFormsInComplexityRange_implementation(
            UsedDimensions const& formerlyUsedDimensions,
            unsigned int minComplexity,
            unsigned int maxComplexity);

    // Tries to fit the points with a form
    // If success, params are stored within the form
    static bool tryFit_implementation(Form& form, unsigned int nbPoints,
                                      Models::ModelConstIterator mBegin, Models::ModelConstIterator mEnd,
                                      unsigned int outputID, unsigned int dimInOutput);

    static Form fitOnePoint_implementation(double t, unsigned int nbDims);

    // Estimates the value for the given inputs
    static std::vector<double> estimate_implementation(Form const& form, std::vector<std::vector<double>> const& x);

    // Returns the form as a readable string (same order as getTerms)
    static std::string formToString_implementation(Form const& form, std::vector<std::string> inputNames);



    // Helper functions

    // Evaluates all terms of the polynomial and returns them as a vector
    static std::vector<double> getTerms(std::vector<double> const& vals, unsigned int degree);

    // Number of terms returned by getTerms
    static unsigned int getNbTerms(unsigned int nbDims, unsigned int degree);

    // Complexity definition
    static unsigned int complexity(unsigned int degree, unsigned int nbUsedDimensions);


    // tryFit() implementations

    // Trying to fit the polynomial to the points with GLPK
    // - the first bool is set to true if a problem occured
    // - the second one is the actual return value
    static std::pair<bool,bool> tryFitGLPK(Form& form, unsigned int nbPoints, Models::ModelConstIterator mBegin, Models::ModelConstIterator mEnd, unsigned int outputID, unsigned int dimInOutput);

    // Same thing with soplex
    static std::pair<bool,bool> tryFitSoplex(Form& form, Models::ModelConstIterator mBegin, Models::ModelConstIterator mEnd, unsigned int outputID, unsigned int dimInOutput);

};



}}



#endif
