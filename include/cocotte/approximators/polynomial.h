#ifndef COCOTTE_APPROXIMATORS_POLYNOMIAL_H
#define COCOTTE_APPROXIMATORS_POLYNOMIAL_H


#include <cocotte/approximators/approximator.h>



namespace Cocotte {
namespace Approximators {



class Polynomial : public Approximator
{

public:

    // Returns the most complex forms under a certain complexity,
    // for each combination of the formerly used dimensions and at most one other.
    virtual std::list<std::list<Form>> getMostComplexForms(
            UsedDimensions const& formerlyUsedDimensions,
            int maxComplexity) override;

    // Tries to fit the points with a form
    // If success, params are stored within the form
    virtual bool tryFit(Form& form, int nbPoints, Models::ModelConstIterator mBegin, Models::ModelConstIterator mEnd, int outputID, int dimInOutput) override;
    virtual bool tryFitGLPK(Form& form, int nbPoints, Models::ModelConstIterator mBegin, Models::ModelConstIterator mEnd, int outputID, int dimInOutput);

    virtual Form fitOnePoint(double t, int nbDims) override;

    // Estimates the value for the given inputs
    virtual std::vector<double> estimate(Form const& form, std::vector<std::vector<double>> const& x) override;

    // Returns the form as a readable string (same order as getTerms)
    virtual std::string formToString(Form const& form, std::vector<std::string> inputNames) override;


private:

    // Evaluates all terms of the polynomial and returns them as a vector
    static std::vector<double> getTerms(std::vector<double> const& vals, int nbDims, int degree);

    // Number of terms returned by getTerms
    static int getNbTerms(int nbDims, int degree);

    // Complexity definition
    static int complexity(int degree, int nbUsedDimensions);

};



}}



#endif
