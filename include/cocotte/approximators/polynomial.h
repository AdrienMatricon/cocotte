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
    std::vector<bool> activeDoF;
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
        archive & form.activeDoF;
        archive & form.params;
    }
};


template <>
struct Fitness<Polynomial>
{
    unsigned int nbDoF; // number of degrees of freedom

    Fitness() = default;
    Fitness(unsigned int nb) : nbDoF(nb){}

    // Relational operators
    // a < b means b is a strictly better fitness
    friend bool operator<(Fitness<Polynomial> const& fitness0, Fitness<Polynomial> const& fitness1)
    {
        return (fitness0.nbDoF > fitness1.nbDoF);
    }

    friend bool operator>(Fitness<Polynomial> const& fitness0, Fitness<Polynomial> const& fitness1)
    {
        return operator<(fitness1, fitness0);
    }

    friend bool operator<=(Fitness<Polynomial> const& fitness0, Fitness<Polynomial> const& fitness1)
    {
        return !(fitness0 > fitness1);
    }

    friend bool operator>=(Fitness<Polynomial> const& fitness0, Fitness<Polynomial> const& fitness1)
    {
        return !(fitness0 < fitness1);
    }


    // Serialization
    template<typename Archive>
    friend void serialize(Archive& archive, Fitness<Polynomial>& fitness, const unsigned int version)
    {
        (void) version; // Unused parameter

        archive & fitness.nbDoF;
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
    // Some information may be stored within the form params
    // The function returns whether is was a success
    static bool tryFit_implementation(
            Form<Polynomial>& form, unsigned int nbPoints,
            Models::ModelConstIterator<Polynomial> const& mBegin,
            Models::ModelConstIterator<Polynomial> const& mEnd,
            unsigned int outputID, unsigned int dimInOutput);

    // Refines a form so as to get the best fitness
    // Params are stored within the form
    // Returns the fitness of that form
    static Fitness<Polynomial> refine_implementation(
            Form<Polynomial>& form, unsigned int nbPoints,
            Models::ModelConstIterator<Polynomial> const& mBegin,
            Models::ModelConstIterator<Polynomial> const& mEnd,
            unsigned int outputID, unsigned int dimInOutput);

    // Tries to fit the points with a form
    // If it succeeds, params are stored within the form
    // The function returns whether is was a success
    //   and (only if it succeeded) the fitness of the form that was found
    //   (the lower the better)
    static std::tuple<bool, Fitness<Polynomial>> tryFitRefine(
            Form<Polynomial>& form, unsigned int nbPoints,
            Models::ModelConstIterator<Polynomial> const& mBegin,
            Models::ModelConstIterator<Polynomial> const& mEnd,
            unsigned int outputID, unsigned int dimInOutput,
            bool refineMode);

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
                                        unsigned int maxDegree,
                                        std::vector<bool> const& isTermActive = std::vector<bool>{});

    // Evaluates the derivatives of those terms with regard to each used dimension
    // The i-th element of the result contains the derivatives of all terms which depended on the i-th value in vals,
    //  in the form of a list of (term ID, derivative) pairs
    static std::vector<std::list<std::pair<unsigned int, double>>> getTermsDerivatives(
            std::vector<double> const& vals,
            unsigned int maxDegree,
            std::vector<bool> const& isTermActive = std::vector<bool>{});

    // Number of terms returned by getTerms()
    static unsigned int getNbTerms(unsigned int nbDims,
                                   unsigned int maxDegree,
                                   std::vector<bool> const& isTermActive = std::vector<bool>{});

    // Complexity definition
    static unsigned int complexity(unsigned int degree, unsigned int nbUsedDimensions);


    // tryFit() implementations

    // Trying to fit the polynomial to the points with GLPK
    // - the first bool is set to true if the call succeeded and false if a problem occured
    // - the other values in the tuple are the actual return values
    // - isTermActive can be used to select polynomial terms and thus remove DoFs
    // - mode affects the solution which is computed and what is put in the form params
    //   - Final: the best solution is computed and the weights are added to the form params
    //   - Initial: same, but the beta params are also added to the form params
    //   - WeightConcentration: the previous beta params are used to bound the optimization
    //                          and the objective becomes minimizing the L1 norm of the weights
    static std::tuple<bool,bool,Fitness<Polynomial>> tryFitGLPK(
            Form<Polynomial>& form, unsigned int nbPoints,
            Models::ModelConstIterator<Polynomial> const& mBegin,
            Models::ModelConstIterator<Polynomial> const& mEnd,
            unsigned int outputID, unsigned int dimInOutput, FitMode mode,
            std::vector<bool> const& isTermActive = std::vector<bool>{});

    // Same thing with soplex
    static std::tuple<bool,bool,Fitness<Polynomial>> tryFitSoplex(
            Form<Polynomial>& form, unsigned int nbPoints,
            Models::ModelConstIterator<Polynomial> const& mBegin,
            Models::ModelConstIterator<Polynomial> const& mEnd,
            unsigned int outputID, unsigned int dimInOutput, FitMode mode,
            std::vector<bool> const& isTermActive = std::vector<bool>{});

};



}}



#endif
