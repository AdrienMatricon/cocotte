#ifndef COCOTTE_APPROXIMATORS_POLYNOMIAL_H
#define COCOTTE_APPROXIMATORS_POLYNOMIAL_H


#include <vector>
#include <utility>
#include <boost/serialization/vector.hpp>
#include <cocotte/useddimensions.h>
#include <cocotte/approximators/forward.h>
#include <cocotte/approximators/approximator.h>



namespace Cocotte {
namespace Approximators {


class Polynomial;


template <>
struct Form<Polynomial>
{
    UsedDimensions usedDimensions;
    UsedDimensions neededDimensions;
    UsedDimensions relevantDimensions;
    unsigned int complexity = 0;
    unsigned int degree = 0;
    std::vector<double> params;

    Form() = default;
    Form(UsedDimensions uD) :
        usedDimensions(uD), neededDimensions(uD), relevantDimensions(uD) {}

    // Serialization
    template<typename Archive>
    friend void serialize(Archive& archive, Form<Polynomial>& form, const unsigned int version)
    {
        (void) version; // Unused parameter

        archive & form.usedDimensions;
        archive & form.neededDimensions;
        archive & form.relevantDimensions;
        archive & form.complexity;
        archive & form.degree;
        archive & form.params;
    }
};


class Polynomial : public Approximator<Polynomial>
{
    friend Approximator<Polynomial>;


private:

    enum class FitMode{
        Initial,
        WeightConcentration,
        Final
    };

    // Computes the minimum complexity such that for each number of dimensions
    //   d in [1, maxNbDimensions] which allows for forms with a complexity <= maxComplexity
    //   there is at least one with a complexity within [minComplexity, maxComplexity]
    static unsigned int getComplexityRangeLowerBound_implementation(
            unsigned int maxNbDimensions,
            unsigned int maxComplexity);

    // Returns all forms of complexity in [minComplexity, maxComplexity],
    // using any number of dimensions from formerlyUsedDimensions
    // using exactly nbNewDimensions dimensions from otherDimensions and not in formerlyUsedDimensions
    // Those forms are returned as a lists of sublists of forms, such that:
    //   - forms in the same sublist have the same complexity
    //   - sublists are sorted by (non strictly) increasing complexity
    static std::list<std::list<Form<Polynomial>>> getFormsInComplexityRange_implementation(
            UsedDimensions const& formerlyUsedDimensions,
            UsedDimensions const& otherDimensions,
            unsigned int nbNewDimensions,
            unsigned int minComplexity,
            unsigned int maxComplexity);

    // Tries to fit the points with a form
    // Some information may be stored within the form params
    // The function returns whether is was a success
    static bool tryFit_implementation(
            Form<Polynomial>& form, unsigned int nbPoints,
            Models::ModelConstIterator<Polynomial> const& mBegin,
            Models::ModelConstIterator<Polynomial> const& mEnd,
            unsigned int outputID, unsigned int dimInOutput);

    static Form<Polynomial> fitOnePoint_implementation(double t, unsigned int nbDims);

    // Estimates the value for the given inputs
    static std::vector<double> estimate_implementation(
            Form<Polynomial> const& form, std::vector<std::vector<double>> const& x);

    // Returns the form as a readable string (same order as getTerms)
    static std::string formToString_implementation(
            Form<Polynomial> const& form, std::vector<std::string> const& inputNames);



    // Helper functions

    // Evaluates all terms of the polynomial and returns them as a vector
    static std::vector<double> getTerms(std::vector<double> const& vals,
                                        unsigned int maxDegree);

    // Number of terms returned by getTerms()
    static unsigned int getNbTerms(unsigned int nbDims,
                                   unsigned int maxDegree);

    // Complexity definition
    static unsigned int complexity(unsigned int degree, unsigned int nbUsedDimensions);


    // tryFit() implementations

    // Trying to fit the polynomial to the points with GLPK
    // - the first bool is set to true if the call succeeded and false if a problem occured
    // - the second bool is whether the points could be fitted
    static std::tuple<bool,bool> tryFitGLPK(
            Form<Polynomial>& form, unsigned int nbPoints,
            Models::ModelConstIterator<Polynomial> const& mBegin,
            Models::ModelConstIterator<Polynomial> const& mEnd,
            unsigned int outputID, unsigned int dimInOutput);

    // Same thing with soplex
    static std::tuple<bool,bool> tryFitSoplex(
            Form<Polynomial>& form, unsigned int nbPoints,
            Models::ModelConstIterator<Polynomial> const& mBegin,
            Models::ModelConstIterator<Polynomial> const& mEnd,
            unsigned int outputID, unsigned int dimInOutput);

};



}}



#endif
