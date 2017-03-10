#ifndef COCOTTE_APPROXIMATORS_POLYNOMIAL_H
#define COCOTTE_APPROXIMATORS_POLYNOMIAL_H


#include <vector>
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
    unsigned int complexity = 0;
    unsigned int degree = 0;
    std::vector<unsigned int> respectiveMaxDegrees;
    std::vector<double> params;

    Form() = default;
    Form(UsedDimensions uD) : usedDimensions(uD) {}

    // Serialization
    template<typename Archive>
    friend void serialize(Archive& archive, Form<Polynomial>& form, const unsigned int version)
    {
        (void) version; // Unused parameter

        archive & form.usedDimensions;
        archive & form.complexity;
        archive & form.degree;
        archive & form.respectiveMaxDegrees;
        archive & form.params;
    }

    // Display
    friend std::ostream& operator<< (std::ostream& out, Form<Polynomial> const& form)
    {
        out << "Complexity " << form.complexity
            << " (degree: " << form.degree << ")"
            << " of dimensions " << form.usedDimensions;

        return out;
    }
};


template <>
struct Fitness<Polynomial>
{
    unsigned int sumDegrees;
    double slack;

    Fitness() = default;
    Fitness(unsigned int nbTerms, double slack) : sumDegrees(nbTerms), slack(slack){}

    // Relational operator
    // a < b means b is a better fitness
    friend bool operator<(Fitness<Polynomial> const& fitness0, Fitness<Polynomial> const& fitness1)
    {
        return (fitness0.sumDegrees > fitness1.sumDegrees)
                || ((fitness0.sumDegrees == fitness1.sumDegrees)
                    && (fitness0.slack > fitness1.slack));
    }

    // Serialization
    template<typename Archive>
    friend void serialize(Archive& archive, Fitness<Polynomial>& fitness, const unsigned int version)
    {
        (void) version; // Unused parameter

        archive & fitness.sumDegrees;
        archive & fitness.slack;
    }
};


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
    static std::list<std::list<Form<Polynomial>>> getFormsInComplexityRange_implementation(
            UsedDimensions const& formerlyUsedDimensions,
            unsigned int nbNewDimensions,
            unsigned int minComplexity,
            unsigned int maxComplexity);

    // Tries to fit the points with a form
    // The function returns whether is was a success
    static bool tryFit_implementation(
            Form<Polynomial>& form, unsigned int nbPoints,
            Models::ModelConstIterator<Polynomial> mBegin,
            Models::ModelConstIterator<Polynomial> mEnd,
            unsigned int outputID, unsigned int dimInOutput);

    // Refines a form so as to get the best fitness
    // Params are stored within the form
    // Returns the fitness of that form
    static Fitness<Polynomial> refine_implementation(
            Form<Polynomial>& form, unsigned int nbPoints,
            Models::ModelConstIterator<Polynomial> mBegin,
            Models::ModelConstIterator<Polynomial> mEnd,
            unsigned int outputID, unsigned int dimInOutput);

    // Tries to fit the points with a form
    // If it succeeds, params are stored within the form
    // The function returns whether is was a success
    //   and (only if it succeeded) the fitness of the form that was found
    //   (the lower the better)
    static std::tuple<bool, Fitness<Polynomial>> tryFitRefine(
            Form<Polynomial>& form, unsigned int nbPoints,
            Models::ModelConstIterator<Polynomial> mBegin,
            Models::ModelConstIterator<Polynomial> mEnd,
            unsigned int outputID, unsigned int dimInOutput,
            bool refineMode);

    static Form<Polynomial> fitOnePoint_implementation(double t, unsigned int nbDims);

    // Estimates the value for the given inputs
    static std::vector<double> estimate_implementation(
            Form<Polynomial> const& form, std::vector<std::vector<double>> const& x);

    // Returns the form as a readable string (same order as getTerms)
    static std::string formToString_implementation(
            Form<Polynomial> const& form, std::vector<std::string> inputNames);



    // Helper functions

    // Evaluates all terms of the polynomial and returns them as a vector
    static std::vector<double> getTerms(std::vector<double> const& vals,
                                        unsigned int maxDegree,
                                        std::vector<unsigned int> const& respectiveMaxDegrees);

    // Number of terms returned by getTerms()
    static unsigned int getNbTerms(unsigned int maxDegree,
                                   std::vector<unsigned int> const& respectiveMaxDegrees);

    // Complexity definition
    static unsigned int complexity(unsigned int degree, unsigned int nbUsedDimensions);


    // tryFit() implementations

    // Trying to fit the polynomial to the points with GLPK
    // - the first bool is set to true if the call succeed and false if a problem occured
    // - the other values in the tuple are the actual return values
    static std::tuple<bool,bool,Fitness<Polynomial>> tryFitGLPK(
            Form<Polynomial>& form, unsigned int nbPoints,
            Models::ModelConstIterator<Polynomial> mBegin,
            Models::ModelConstIterator<Polynomial> mEnd,
            unsigned int outputID, unsigned int dimInOutput,
            std::vector<unsigned int> const& respectiveMaxDegrees);

    // Same thing with soplex
    static std::tuple<bool,bool,Fitness<Polynomial>> tryFitSoplex(
            Form<Polynomial>& form,
            Models::ModelConstIterator<Polynomial> mBegin,
            Models::ModelConstIterator<Polynomial> mEnd,
            unsigned int outputID, unsigned int dimInOutput,
            std::vector<unsigned int> const& respectiveMaxDegrees);

};



}}



#endif
