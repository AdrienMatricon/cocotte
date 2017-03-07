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
    //   d in [1, maxNbDimensions] which allows for forms with a complexity <= maxComplexity
    //   there is at least one with a complexity within [minComplexity, maxComplexity]
    static unsigned int getComplexityRangeLowerBound_implementation(
            unsigned int maxNbDimensions,
            unsigned int maxComplexity);

    // Returns all forms of complexity in [minComplexity, maxComplexity],
    // using exactly nbNewDimensions dimensions not in formerlyUsedDimensions
    // Those forms are returned as a lists of sublists of forms, such that:
    //   - forms in the same sublist have the same complexity
    //   - sublists are sorted by (non strictly) increasing complexity
    static std::list<std::list<Form>> getFormsInComplexityRange_implementation(
            UsedDimensions const& formerlyUsedDimensions,
            unsigned int nbNewDimensions,
            unsigned int minComplexity,
            unsigned int maxComplexity);

    // Tries to fit the points with a form
    // If it succeeds, params are stored within the form
    // The function returns whether is was a success
    //   and (only if it succeeded) the fitness of the form that was found
    //   (the lower the better)
    static std::tuple<bool,double> tryFit_implementation(Form& form, unsigned int nbPoints,
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
    // - the other values in the tuple are the actual return values
    static std::tuple<bool,bool,double> tryFitGLPK(Form& form, unsigned int nbPoints,
                                                   Models::ModelConstIterator mBegin, Models::ModelConstIterator mEnd,
                                                   unsigned int outputID, unsigned int dimInOutput);

    // Same thing with soplex
    static std::tuple<bool,bool,double> tryFitSoplex(Form& form,
                                                     Models::ModelConstIterator mBegin, Models::ModelConstIterator mEnd,
                                                     unsigned int outputID, unsigned int dimInOutput);

};



}}



#endif
