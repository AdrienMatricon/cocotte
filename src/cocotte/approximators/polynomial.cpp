
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
#include <fstream>
using std::ofstream;
#include <string>
using std::string;
using std::to_string;
#include <vector>
using std::vector;
#include <list>
using std::list;
#include <cmath>
using std::abs;
using std::exp;
#include <algorithm>
using std::min;
using std::max;
#include <utility>
using std::pair;
using std::make_pair;
#include <tuple>
using std::tuple;
using std::make_tuple;
#include <exception>
#include <glpk.h>
#include <soplex/src/soplex.h>
#include <cocotte/datatypes.h>
using Cocotte::DataPoint;
#include <cocotte/approximators/polynomial.h>
using FormType = Cocotte::Approximators::Form<Cocotte::Approximators::Polynomial>;



namespace Cocotte {
namespace Approximators {


// Computes the minimum complexity such that for each number of dimensions
//   d in [1, maxNbDimensions] which allows for forms with a complexity <= maxComplexity
//   there is at least one with a complexity within [minComplexity, maxComplexity]
unsigned int Polynomial::getComplexityRangeLowerBound_implementation(
        unsigned int maxNbDimensions,
        unsigned int maxComplexity)
{
    if (maxComplexity == 1)
    {
        return 1u;
    }

    unsigned int minComplexity = maxComplexity;

    for (unsigned int d = 1; d < maxNbDimensions; ++d)
    {
        unsigned int N = 1;                 // degree
        unsigned int c = complexity(N, d);

        // We look for the highest possible degree N >= 1 that allows for a complexity
        // under maxComplexity with d variables
        {
            // There is no N >= 1 that can work
            if (c > maxComplexity)
            {
                break;
            }

            // We increment N as much as possible
            while (c < maxComplexity)
            {
                c = complexity(++N, d);
            }

            // It's okay if c == maxComplexity
            // Otherwise we decrement N
            if (c > maxComplexity)
            {
                c = complexity(--N, d);
            }
        }

        minComplexity = min(minComplexity, c);
    }

    return minComplexity;
}


// Returns all forms of complexity in [minComplexity, maxComplexity],
// using any number of dimensions from formerlyUsedDimensions
// using exactly nbNewDimensions dimensions from otherDimensions and not in formerlyUsedDimensions
// Those forms are returned as a lists of sublists of forms, such that:
//   - forms in the same sublist have the same complexity
//   - sublists are sorted by (non strictly) increasing complexity
list<list<FormType>> Polynomial::getFormsInComplexityRange_implementation(
        UsedDimensions const& formerlyUsedDimensions,
        UsedDimensions const& otherDimensions,
        unsigned int nbNewDimensions,
        unsigned int minComplexity,
        unsigned int maxComplexity)
{
    UsedDimensions newDimensions = otherDimensions ^ formerlyUsedDimensions.complement();

    // Trivial cases
    if ( (minComplexity > maxComplexity)
         || (newDimensions.getNbUsed() < nbNewDimensions) )
    {
        return {};
    }

    unsigned int const nbAvailableDimensions = formerlyUsedDimensions.getNbUsed() + nbNewDimensions;

    if (formerlyUsedDimensions.getTotalNbDimensions() < nbAvailableDimensions)
    {
        return {};
    }

    unsigned int const maxNbDims = min(maxComplexity - 1u,      // c = N*d+1, thus d <= c-1 for N > 0
                                       nbAvailableDimensions);


    // Now onto the normal case
    list<list<FormType>> result{};

    // First the form which uses 0 dimensions
    if ((minComplexity <= 1u) && (nbNewDimensions == 0u))
    {
        // N = 0 or d = 0, which is equivalent
        FormType form(UsedDimensions(formerlyUsedDimensions.getTotalNbDimensions()));   // d = 0
        form.degree = 0u; // N = 0
        form.complexity = 1u;

        result.push_back(list<FormType>{form});
    }

    // Then the other forms
    for (unsigned int d = max(1u, nbNewDimensions); d <= maxNbDims; ++d)
    {
        // We start from the minimum N such that the complexity
        // is in [minComplexity, maxComplexity]
        unsigned int N = 1, c = complexity(N, d);
        while (c < minComplexity)
        {
            ++N;
            c = complexity(N, d);
        }

        // Then we iterate over N as long as the complexity
        // stays in [minComplexity, maxComplexity]
        for (;c <= maxComplexity; c = complexity(++N, d))
        {
            // We compute all combinations of nbNewDimensions newly used dimensions
            // and (d - nbNewDimensions) formerly used ones
            list<UsedDimensions> const dimensionCombinations
                    = UsedDimensions::getMixedCombinations(
                        formerlyUsedDimensions, d - nbNewDimensions,
                        newDimensions, nbNewDimensions);

            // We compute all corresponding forms and create a sublist to store them
            list<FormType> newForms;
            for (auto& comb : dimensionCombinations)
            {
                FormType form(std::move(comb));
                form.complexity = c;
                form.degree = N;
                newForms.push_back(std::move(form));
            }

            // We insert the new sublist in the list,
            // in such a way that it stays sorted by increasing complexity
            auto rIt = result.begin();
            auto const rEnd = result.end();

            while ((rIt != rEnd) && (c > rIt->back().complexity))
            {
                ++rIt;
            }

            result.insert(rIt, std::move(newForms));
        }
    }

    return result;
}


// Tries to fit the points with a form
// Some information may be stored within the form params
// The function returns whether is was a success
bool Polynomial::tryFit_implementation(
        Form<Polynomial>& form, unsigned int nbPoints,
        Models::ModelConstIterator<Polynomial> const& mBegin,
        Models::ModelConstIterator<Polynomial> const& mEnd,
        unsigned int outputID, unsigned int dimInOutput)
{
    auto const tryFitRefineResult = tryFitRefine(form, nbPoints, mBegin, mEnd, outputID, dimInOutput, false);
    return std::get<0>(tryFitRefineResult);
}

// Refines a form so as to get the best fitness
// Returns the fitness of that form
Fitness<Polynomial> Polynomial::refine_implementation(
        Form<Polynomial>& form, unsigned int nbPoints,
        Models::ModelConstIterator<Polynomial> const& mBegin,
        Models::ModelConstIterator<Polynomial> const& mEnd,
        unsigned int outputID, unsigned int dimInOutput)
{
    auto const tryFitRefineResult = tryFitRefine(form, nbPoints, mBegin, mEnd, outputID, dimInOutput, true);
    return std::get<1>(tryFitRefineResult);
}

// Tries to fit the points with a form
// If it succeeds, params are stored within the form
// The function returns whether is was a success
//   and (only if it succeeded) the fitness of the form that was found
tuple<bool, Fitness<Polynomial>> Polynomial::tryFitRefine(
        FormType& form, unsigned int nbPoints,
        Models::ModelConstIterator<Polynomial> const& mBegin,
        Models::ModelConstIterator<Polynomial> const& mEnd,
        unsigned int outputID, unsigned int dimInOutput,
        bool refineMode)
{
    vector<bool> activeDoF;         // Default value when not refining solutions
    unsigned int const nbUsedInputDims = form.usedDimensions.getNbUsed();
    unsigned int const degree = form.degree;


    // Lambda function to wrap calls to soplex and GLPK
    auto lambda = [&](FormType& candidateForm, FitMode mode)
    {
        {
            tuple<bool, Fitness<Polynomial>> result;
            auto const soplexResult = tryFitSoplex(
                        candidateForm, nbPoints, mBegin, mEnd, outputID, dimInOutput, mode, activeDoF);

            if (std::get<0>(soplexResult))
            {
                return make_tuple(std::get<1>(soplexResult), std::get<2>(soplexResult));
            }
            else        // Something went wrong when calling soplex
            {
                auto const glpkResult = tryFitGLPK(
                            candidateForm, nbPoints, mBegin, mEnd, outputID, dimInOutput, mode, activeDoF);

                if (std::get<0>(glpkResult))
                {
                    return make_tuple(std::get<1>(glpkResult), std::get<2>(glpkResult));
                }
                else    // Something went wrong while calling GLPK
                {
                    throw std::runtime_error("Neither soplex nor GLPK can solve the linear program");
                }
            }
            return result;
        }
    };


    // If it's a call to tryFit:
    // - we compute normalization coefficients and store them in form.params
    // - we also compute the amplitude of the output and store it in form.params
    // - we try to fit the points, and return the result
    if (!refineMode)
    {
        auto const usedInputDimIds = form.usedDimensions.getIds();
        unsigned int const totalNbInputDims = form.usedDimensions.getTotalNbDimensions();

        // We determine the maximum absolute value for each used dimension and the output
        // (note: the vectors are of size totalNbInputDims because we want maxes[i] to be the max for dimension i)
        vector<double> maxes(totalNbInputDims), inputNormalizationFactors(totalNbInputDims);
        double maxOutput = 0.;

        for (auto mIt = mBegin; mIt != mEnd; ++mIt)
        {
            for (auto const& id : usedInputDimIds)
            {
                double const val = abs(mIt->x[id].value);
                if (val > maxes[id])
                {
                    maxes[id] = val;
                }
            }

            double const val = abs(mIt->t[outputID][dimInOutput].value);
            if (val > maxOutput)
            {
                maxOutput = val;
            }
        }


        // We compute the normalization coefficients
        form.params.clear();
        form.params.reserve(nbUsedInputDims+1);

        for (auto const& id : usedInputDimIds)
        {
            if (maxes[id] > 0.)
            {
                inputNormalizationFactors[id] = 1./maxes[id];
            }
            form.params.push_back(inputNormalizationFactors[id]);
        }

        form.params.push_back(maxOutput);


        // We determine if we can fit the points
        tuple<bool, Fitness<Polynomial>> result;

        if (getNbTerms(nbUsedInputDims, degree, activeDoF) > nbPoints)
        {
            std::get<0>(result) = true;
        }
        else
        {
            result = lambda(form, FitMode::Initial);
        }

        return result;
    }



    // If we are in refine mode, tryFit has already been called and the form
    //  should be able to fit the points; now we have to see if we can remove some terms
    // We try to find a solution in which as many DoF as possible correspond to params set to zero
    tuple<bool, Fitness<Polynomial>> result;
    unsigned int const nbNormalizationParams = form.usedDimensions.getNbUsed()+1;
    unsigned int const nbDoF = getNbTerms(nbUsedInputDims, degree, activeDoF);

    // We compute a solution if it has not already been done
    if (form.params.size() == nbNormalizationParams)
    {
        result = lambda(form, FitMode::Initial);

        if (!std::get<0>(result))
        {
            string message = "Cannot fit the points somehow when it should be possible.";
            message += " This should not happen.";
            throw std::runtime_error(message);
        }
    }

    // Now we concentrate the weights so that we can remove DoFs
    if (nbDoF > 1)
    {
        lambda(form, FitMode::WeightConcentration);

        // We determine a cutoff to determine which weights are at 0
        double average = 0;
        for (unsigned int i = nbNormalizationParams; i < nbNormalizationParams + nbDoF; ++i)
        {
            average += abs(form.params[i]);
        }
        average /= nbDoF;

        double const cutoff = 1.e-10 * average;

        // We disable all DoFs under that cutoff
        activeDoF.resize(nbDoF, true);
        unsigned int nbDisabledDoF = 0;
        for (unsigned int i = nbNormalizationParams, j = 0; j < nbDoF; ++i, ++j)
        {
            if (abs(form.params[i]) <= cutoff)
            {
                activeDoF[j] = false;
                ++nbDisabledDoF;
            }
        }

        // Edge case
        if (nbDisabledDoF == nbDoF)
        {
            // All params are 0, we at least keep the constant term
            activeDoF[0] = true;
        }
    }


    // We remove the DoFs whose params have been set to zero
    //  and check that we can still find a solution
    result = lambda(form, FitMode::Final);

    while (!std::get<0>(result))
    {
        // Removed one too many DoF because of floating point precision, adding one back
        auto const aEnd = activeDoF.end();
        for (auto aIt = activeDoF.begin(); aIt != aEnd; ++aIt)
        {
            if (!*aIt)
            {
                *aIt = true;
                break;
            }
        }

        // We do the slack variable minimization again
        result = lambda(form, FitMode::Final);
    }

    form.activeDoF = activeDoF;

    return result;
}


FormType Polynomial::fitOnePoint_implementation(double t, unsigned int nbDims)
{
    FormType form (UsedDimensions{nbDims});
    form.complexity = 1;
    form.degree = 0;
    form.params = vector<double>{t,1};

    return form;
}


// Estimates the value for the given inputs
vector<double> Polynomial::estimate_implementation(FormType const&form, vector<vector<double>> const& points)
{
    unsigned int const nbPoints = points.size();
    unsigned int const nbDims = form.usedDimensions.getNbUsed();
    unsigned int const degree = form.degree;

    auto const pBegin = form.params.begin();

    vector<double> result;
    result.reserve(nbPoints);

    vector<vector<double>> processedPoints;

    // Selecting dimensions and normalizing
    {
        processedPoints.reserve(nbPoints);
        list<unsigned int> const& dimIds = form.usedDimensions.getIds();
        auto const dBegin = dimIds.begin(), dEnd = dimIds.end();

        for (unsigned int i = 0; i < nbPoints; ++i)
        {
            vector<double> const& currentPoint = points[i];
            vector<double> currentProcessed;
            currentProcessed.reserve(nbDims);
            auto pIt = pBegin;

            for (auto dIt = dBegin; dIt != dEnd; ++dIt, ++pIt)
            {
                currentProcessed.push_back(*pIt * currentPoint[*dIt]);
            }

            processedPoints.push_back(currentProcessed);
        }
    }

    double const maxTrainingT = form.params[nbDims];
    auto const pCoeffsBegin = pBegin + nbDims + 1;

    // Computing the weighted sum of the terms of the polynomial
    for (auto& point : processedPoints)
    {
        vector<double> const terms = Polynomial::getTerms(point, degree, form.activeDoF);
        double t = 0;
        auto pIt = pCoeffsBegin;

        for (auto& term : terms)
        {
            t += *pIt * term;
            ++pIt;
        }

        result.push_back(maxTrainingT * t);
    }

    return result;
}


// Returns the form as a readable string (same order as getTerms)
string Polynomial::formToString_implementation(FormType const& form, vector<string> const& inputNames)
{
    unsigned int const nbDims = form.usedDimensions.getNbUsed();
    unsigned int const degree = form.degree;

    // First we select the variable names
    vector<string> dimNames;
    dimNames.reserve(nbDims);

    for (auto& dim : form.usedDimensions.getIds())
    {
        dimNames.push_back(inputNames[dim]);
    }


    // Then we generate the string
    string result = "c = " + to_string(form.complexity);
    result += ": N = " + to_string(degree);
    result += ", d = " + to_string(nbDims);

    if (nbDims > 0)
    {
        result +=  " (" + dimNames[0];

        for (unsigned int i = 1; i < nbDims; ++i)
        {
            result += "," + dimNames[i];
        }

        result += ")";
    }

    return result;
}




// Helper functions


// Evaluates all terms of the polynomial and returns them as a vector
vector<double> Polynomial::getTerms(vector<double> const& vals,
                                    unsigned int maxDegree,
                                    vector<bool> const& isTermActive)
{
    // We handle the trivial case first
    if (maxDegree < 1)
    {
        // If the degree is zero, we return a vector only containing a 1
        return vector<double>(1,1);
    }

    // Then normal cases
    vector<list<double>> terms;
    auto const vBegin = vals.begin(), vEnd = vals.end();

    // We initialize terms by adding the first dimension terms to it
    {
        terms.reserve(maxDegree + 1);
        double const val = *vBegin;
        double temp = val;

        terms.push_back(list<double>(1, 1.));
        terms.push_back(list<double>(1, temp));

        unsigned int N;
        for (N = 2; N <= maxDegree; ++N)
        {
            temp *= val;
            terms.push_back(list<double>(1, temp));
        }
    }

    // We now add all other dimensions
    for (auto vIt = vBegin + 1; vIt != vEnd; ++vIt)
    {
        auto newTerms = terms;

        double const val = *vIt;
        double temp = val;

        // We add all terms of degree 1 in the new dimension
        {
            newTerms[1].push_back(temp);

            unsigned int const endLoop = maxDegree;
            for (unsigned int M = 1; M < endLoop; ++M)
            {
                // M is the degree in the old dimensions
                unsigned int const newDeg = M + 1;
                for (auto& term : terms[M])
                {
                    newTerms[newDeg].push_back(term * temp);
                }
            }
        }

        // We add all other terms
        for (unsigned int N = 2; N <= maxDegree; ++N)
        {
            // N is the degree in the new dimension
            temp *= val;
            newTerms[N].push_back(temp);

            unsigned int const endLoop = maxDegree - N + 1;
            for (unsigned int M = 1; M < endLoop; ++M)
            {
                // M is the degree in the old dimensions
                unsigned int const newDeg = M + N;
                for (auto& term : terms[M])
                {
                    newTerms[newDeg].push_back(term * temp);
                }
            }
        }

        terms = std::move(newTerms);
    }

    // We put every term in a vector and return it
    vector<double> result;

    for (auto& someTerms : terms)
    {
        result.insert(result.end(), someTerms.begin(), someTerms.end());
    }

    if (!isTermActive.empty())
    {
        // We filter the inactive terms out
        auto it = isTermActive.begin();
        result.erase(std::remove_if(result.begin(), result.end(),
                                    [&](double){return !(*(it++));}),
                     result.end());
    }

    return result;
}


// Evaluates the derivatives of those terms with regard to each used dimension
// The i-th element of the result contains the derivatives of all terms which depended on the i-th value in vals,
//  in the form of a list of (term ID, derivative) pairs
vector<list<pair<unsigned int, double>>> Polynomial::getTermsDerivatives(
        vector<double> const& vals,
        unsigned int maxDegree,
        vector<bool> const& isTermActive)
{
    // We evacuate the trivial case
    if (maxDegree < 1)
    {
        return vector<list<pair<unsigned int, double>>>(vals.size());
    }

    // Onto the real thing
    // First we list the terms, in the same order as getTerms(),
    //  determining which term has which degree in what dimension
    unsigned int const nbDims = vals.size();
    vector<vector<unsigned int>> termsInOrder;
    {
        vector<list<vector<unsigned int>>> terms;
        // We initialize terms by adding the first dimension terms to it
        {
            terms.reserve(maxDegree + 1);

            vector<unsigned int> const degreeZeroTerm(nbDims, 0);

            terms.push_back(list<vector<unsigned int>>{degreeZeroTerm});

            unsigned int N;
            for (N = 1; N <= maxDegree; ++N)
            {
                auto temp = degreeZeroTerm;
                temp[0] = N;
                terms.push_back(list<vector<unsigned int>>{temp});
            }
        }

        // We now add all other dimensions
        for (unsigned int dimID = 1; dimID < nbDims; ++dimID)
        {
            auto newTerms = terms;
            for (unsigned int N = 1; N <= maxDegree; ++N)
            {
                // N is the degree in the new dimension
                unsigned int const endLoop = maxDegree - N + 1;
                for (unsigned int M = 0; M < endLoop; ++M)
                {
                    // M is the degree in the old dimensions
                    unsigned int const newDeg = M + N;
                    for (auto& term : terms[M])
                    {
                        auto temp = term;
                        temp[dimID] += N;
                        newTerms[newDeg].push_back(temp);
                    }
                }
            }

            terms = std::move(newTerms);
        }

        // We put every term in a vector
        for (auto& someTerms : terms)
        {
            termsInOrder.insert(termsInOrder.end(), someTerms.begin(), someTerms.end());
        }

        if (!isTermActive.empty())
        {
            // We filter the inactive terms out
            auto it = isTermActive.begin();
            termsInOrder.erase(std::remove_if(termsInOrder.begin(), termsInOrder.end(),
                                              [&](vector<unsigned int>){return !(*(it++));}),
                               termsInOrder.end());
        }
    }


    // We compute the powers of the values so that we don't recompute them all the time
    vector<vector<double>> powers(nbDims, vector<double>(maxDegree + 1, 1.));
    for (unsigned int i = 0; i < nbDims; ++i)
    {
        double const val = vals[i];
        double temp = val;
        powers[i][1] = temp;

        for (unsigned int j = 2; j <= maxDegree; ++j)
        {
            temp *= val;
            powers[i][j] = temp;
        }
    }


    // Now it's time to compute the derivatives
    vector<list<pair<unsigned int, double>>> result(nbDims);
    unsigned int const nbTerms = termsInOrder.size();
    for (unsigned int termID = 0; termID < nbTerms; ++termID)
    {
        auto const term = termsInOrder[termID];
        for (unsigned int i = 0; i < nbDims; ++i)
        {
            unsigned int const degInI = term[i];
            if (degInI > 0)
            {
                double derivative = 1.;

                if (degInI > 1)
                {
                    derivative = degInI * powers[i][degInI - 1];
                }

                for (unsigned int j = 0; j < nbDims; ++j)
                {
                    unsigned int const degInJ = term[j];
                    if ((j != i) && (degInJ > 0))
                    {
                        derivative *= powers[j][degInJ];
                    }
                }

                result[i].push_back(make_pair(termID, derivative));
            }
        }
    }

    return result;
}


// Number of terms returned by getTerms
unsigned int Polynomial::getNbTerms(unsigned int nbDims,
                                   unsigned int maxDegree,
                                    vector<bool> const& isTermActive)
{
    if (isTermActive.empty())
    {
        // There are nbDims among (maxDegree + nbDims) terms
        unsigned int numerator = 1;
        unsigned int denominator = 1;
        unsigned int const sum = nbDims + maxDegree;

        for (unsigned int i = 0; i < nbDims; ++i)
        {
            numerator *= sum - i;
            denominator *= nbDims - i;
        }

        return numerator / denominator;
    }


    // Terms are sorted between active and inactive
    unsigned int result = 0;
    for (auto const& active : isTermActive)
    {
        if (active)
        {
            ++result;
        }
    }

    return result;
}



// Complexity definition
unsigned int Polynomial::complexity(unsigned int degree, unsigned int nbUsedDimensions)
{
    return (degree * nbUsedDimensions) + 1;
}



// Trying to fit the polynomial to the points with GLPK
// - the first bool is set to true if the call succeed and false if a problem occured
// - the other values in the tuple are the actual return values
// - isTermActive can be used to select polynomial terms and thus remove DoFs
// - mode affects the solution which is computed and what is put in the form params
//   - Final: the best solution is computed and the weights are added to the form params
//   - Initial: same, but the beta params are also added to the form params
//   - WeightConcentration: the previous beta params are used to bound the optimization
//                          and the objective becomes minimizing the L1 norm of the weights
tuple<bool,bool,Fitness<Polynomial>> Polynomial::tryFitGLPK(
        FormType& form,unsigned int nbPoints,
        Models::ModelConstIterator<Polynomial> const& mBegin,
        Models::ModelConstIterator<Polynomial> const& mEnd,
        unsigned int outputID, unsigned int dimInOutput, FitMode mode,
        vector<bool> const& isTermActive)
{
    auto const usedInputDimIds = form.usedDimensions.getIds();
    unsigned int const nbUsedInputDims = form.usedDimensions.getNbUsed();
    unsigned int const totalNbInputDims = form.usedDimensions.getTotalNbDimensions();
    unsigned int const degree = form.degree;
    unsigned int const nbTerms = getNbTerms(nbUsedInputDims, degree, isTermActive);

    // Extracting the normalization factors
    vector<double> inputNormalizationFactors(totalNbInputDims);

    auto pIt = form.params.begin();
    for (auto const& id : usedInputDimIds)
    {
        inputNormalizationFactors[id] = *pIt;
        ++pIt;
    }

    double const maxOutput = form.params[nbUsedInputDims];
    double outputNormalizationFactor = 0.;
    if (maxOutput > 0.)
    {
        outputNormalizationFactor = 1./maxOutput;
    }


    // Let's create the optimization problem
    //
    // Naming convention:
    //    - i the index over points
    //    - j the index over polynomial terms
    //    - k the index over dimensions
    //
    // Optimization parameters:
    //    - 1 param per term (weight_j)
    //    - 1 param per point (xi_i) for the normalized distance to the approximation
    //    - 1 param (alpha) for the shared part of the xi_i which is due to the imprecision
    //      on the output values
    //    - 1 param per xi_i (i.e. per point) and per input dimension (beta_ik)
    //      for the part of those xi_i which is due to the imprecision on the input values
    //    - In terms of priorities,
    //      we want to minimize all xi_i first, then all beta_ik
    // When mode is WeightMinimization:
    //    - 1 param per term (gamma_j) is added to bound the absolute values of the weights
    //    - the beta_ik are bounded by their values in the previous call
    //    - what we minimize is the sum of the gamma_j,
    //      using values from the previous call to weigh their importance
    glp_prob *lp;
    lp = glp_create_prob();
    glp_set_obj_dir(lp, GLP_MIN);

    unsigned int const nbOptimParameters =
            nbTerms                                 // weight_i
            + nbPoints                              // xi_i
            + 1                                     // alpha
            + nbUsedInputDims*nbPoints              // beta_ik
            + ((mode == FitMode::WeightConcentration)?
                nbTerms : 0);                       // gamma_i

    unsigned int const firstWeightParamID = 1;  // IDs start at 1 in GLPK
    unsigned int const firstXiParamID = firstWeightParamID + nbTerms;
    unsigned int const alphaParamID = firstXiParamID + nbPoints;
    unsigned int const firstBetaParamID = alphaParamID + 1;
    unsigned int const firstGammaParamID = firstBetaParamID + nbPoints*nbUsedInputDims;

    // Optimization parameters declaration
    {
        // Coefficients in what we minimize
        double xiCoeff = 0.;
        double alphaCoeff = 0.5;
        double betaCoeff = 1.;

        if (mode == FitMode::WeightConcentration)
        {
            xiCoeff = 0.;
            alphaCoeff = 0.;
            betaCoeff = 0.;
        }


        glp_add_cols(lp, nbOptimParameters);

        // weight_j (unbounded)
        for (unsigned int j = 0; j < nbTerms; ++j)
        {
            glp_set_col_bnds(lp, firstWeightParamID + j, GLP_FR, 0.0, 0.0);
            glp_set_obj_coef(lp, firstWeightParamID + j, 0.0);
        }

        // xi_i (positive)
        for (unsigned int i = 0; i < nbPoints; ++i)
        {
            glp_set_col_bnds(lp, firstXiParamID + i, GLP_LO, 0.0, 0.0);
            glp_set_obj_coef(lp, firstXiParamID + i, xiCoeff);
        }

        // alpha (in [0,1])
        glp_set_col_bnds(lp, alphaParamID, GLP_DB, 0.0, 1.0);
        glp_set_obj_coef(lp, alphaParamID, alphaCoeff);

        // beta_ik (positive)
        if (mode == FitMode::WeightConcentration)
        {
            unsigned int const nbBetas = nbUsedInputDims*nbPoints;
            unsigned int const firstBetaInFormParams = form.params.size() - nbBetas;
            for (unsigned int i = 0; i < nbPoints; ++i)
            {
                unsigned int const firstBetaIParamID = firstBetaParamID + i*nbUsedInputDims;
                unsigned int const firstBetaInFormParamsForThisI = firstBetaInFormParams + i*nbUsedInputDims;

                for (unsigned int k = 0; k < nbUsedInputDims; ++k)
                {
                    glp_set_col_bnds(lp, firstBetaIParamID + k, GLP_DB, 0.0, 1.1*
                                     form.params[firstBetaInFormParamsForThisI+k]);
                    glp_set_obj_coef(lp, firstBetaIParamID + k, betaCoeff);
                }
            }
        }
        else
        {
            for (unsigned int i = 0; i < nbPoints; ++i)
            {
                unsigned int const firstBetaIParamID = firstBetaParamID + i*nbUsedInputDims;
                for (unsigned int k = 0; k < nbUsedInputDims; ++k)
                {
                    glp_set_col_bnds(lp, firstBetaIParamID + k, GLP_LO, 0.0, 0.0);
                    glp_set_obj_coef(lp, firstBetaIParamID + k, betaCoeff);
                }
            }
        }

        // gamma_j (positive)
        if (mode == FitMode::WeightConcentration)
        {
            // We compute the sum of all the absolute values of the weights
            //  in the solution from the previous call
            unsigned int const firstWeightInFormParams = nbUsedInputDims + 1;
            vector<double> absoluteWeights;
            absoluteWeights.reserve(nbTerms);

            for (unsigned int j = 0; j < nbTerms; ++j)
            {
                absoluteWeights.push_back(abs(form.params[firstWeightInFormParams+j]));
            }

            double sumAbsWeights = 0.;
            for (auto const& w : absoluteWeights)
            {
                sumAbsWeights += w;
            }

            // The higher the weight a term had in the solution from the previous call,
            //  the lower the cost we place on it, so that weights are concentrated
            for (unsigned int j = 0; j < nbTerms; ++j)
            {
                double const gammaCoeff = sumAbsWeights - absoluteWeights[j];
                glp_set_col_bnds(lp, firstGammaParamID + j, GLP_LO, 0.0, 0.0);
                glp_set_obj_coef(lp, firstGammaParamID + j, gammaCoeff);
            }
        }
    }

    // Number of constraints and possible non-zero coefficients in those constraints
    unsigned int nbConstraints =
            nbPoints * (
                1                                       // bounds on xi_i
                + 2                                     // bounds on polynomial values
                + 2*nbUsedInputDims                     // bounds on beta_ik
                )
            + ((mode == FitMode::WeightConcentration)?
                2*nbTerms:0);                           // bounds on weights

    unsigned int maxNbNonZeroCoeffs =
            nbPoints * (
                (2 + nbUsedInputDims)                   // bounds on xi_i
                + 2 * (1 + nbTerms)                     // bounds on polynomial values
                + 2 * nbUsedInputDims * (1 + nbTerms)   // bounds on beta_ik
                )
            + ((mode == FitMode::WeightConcentration)?
              4*nbTerms:0);                             // bounds on weights

    vector<int> iMat(1+maxNbNonZeroCoeffs);             // Coefficient constraint IDs
    vector<int>  jMat(1+maxNbNonZeroCoeffs);            // Coefficient variable IDs
    vector<double> cMat(1+maxNbNonZeroCoeffs);          // Coefficient values

    // Optimization constraints declaration
    {

        glp_add_rows(lp, nbConstraints);
        int constraintID = 1;
        int coeffID = 1;


        if (mode == FitMode::WeightConcentration)
        {
            // Constraints over the weights
            for (unsigned int j = 0; j < nbTerms; ++j)
            {
                // w_j <= gamma_j
                // => w_j - gamma_j <= 0
                {
                    iMat[coeffID] = constraintID;
                    jMat[coeffID] = firstWeightParamID + j;
                    cMat[coeffID] = 1.;                                                         // w_j
                    ++coeffID;

                    iMat[coeffID] = constraintID;
                    jMat[coeffID] = firstGammaParamID + j;
                    cMat[coeffID] = -1.;                                                        // -gamma_j
                    ++coeffID;

                    glp_set_row_bnds(lp, constraintID, GLP_UP, 0., 0.);                         // <= 0
                    ++constraintID;
                }

                // -w_j <= gamma_j
                // => -w_j - gamma_j <= 0
                {
                    iMat[coeffID] = constraintID;
                    jMat[coeffID] = firstWeightParamID + j;
                    cMat[coeffID] = -1.;                                                        // -w_j
                    ++coeffID;

                    iMat[coeffID] = constraintID;
                    jMat[coeffID] = firstGammaParamID + j;
                    cMat[coeffID] = -1.;                                                        // -gamma_j
                    ++coeffID;

                    glp_set_row_bnds(lp, constraintID, GLP_UP, 0., 0.);                         // <= 0
                    ++constraintID;
                }
            }
        }

        // Constraints over each point
        unsigned int i = 0;
        for (auto mIt = mBegin; mIt != mEnd; ++mIt, ++i)
        {
            unsigned int firstBetaIParamID = firstBetaParamID + i*nbUsedInputDims;

            // Normalized values and precision for the output
            auto const normalizedOutputVal = mIt->t[outputID][dimInOutput].value * outputNormalizationFactor;
            auto const normalizedOutputPrec = mIt->t[outputID][dimInOutput].precision * outputNormalizationFactor;

            // Normalized input
            vector<double> normalizedInput, normalizedInputPrecisions;
            normalizedInput.reserve(nbUsedInputDims);
            normalizedInputPrecisions.reserve(nbUsedInputDims);

            for (auto const& id : usedInputDimIds)
            {
                normalizedInput.push_back(mIt->x[id].value * inputNormalizationFactors[id]);
                normalizedInputPrecisions.push_back(mIt->x[id].precision * inputNormalizationFactors[id]);
            }


            // On xi_i
            {
                // xi_i <= alpha + sum_k(beta_ik)
                // => - xi_i + alpha + sum_k(beta_ik) >= 0
                iMat[coeffID] = constraintID;
                jMat[coeffID] = firstXiParamID + i;
                cMat[coeffID] = -1.;                                                        // -xi_i
                ++coeffID;

                iMat[coeffID] = constraintID;
                jMat[coeffID] = alphaParamID;
                cMat[coeffID] = 1.;                                                         // alpha
                ++coeffID;

                for (unsigned int k = 0; k < nbUsedInputDims; ++k)
                {
                    iMat[coeffID] = constraintID;
                    jMat[coeffID] = firstBetaIParamID + k;
                    cMat[coeffID] = 1.;                                                     // beta_ik
                    ++coeffID;
                }

                glp_set_row_bnds(lp, constraintID, GLP_LO, 0., 0.);                         // >= 0
                ++constraintID;
            }


            // On the values of the polynomial
            {
                vector<double> const terms = getTerms(normalizedInput, degree, isTermActive);

                // (f(x_i) - t_i) <= outputPrec_i * xi_i
                // => f(x_i) - outputPrec_i * xi_i <= t_i
                // where f(x_i) = sum_j(weight_j * f_j(x_i))
                // and f_j the j-th polynomial term
                {
                    for (unsigned int j = 0; j < nbTerms; ++j)
                    {
                        iMat[coeffID] = constraintID;
                        jMat[coeffID] = firstWeightParamID + j;
                        cMat[coeffID] = terms[j];                                           // weight_j * f_j
                        ++coeffID;
                    }

                    iMat[coeffID] = constraintID;
                    jMat[coeffID] = firstXiParamID + i;
                    cMat[coeffID] = -normalizedOutputPrec;                                  // -outputPrec_i * xi_i
                    ++coeffID;

                    glp_set_row_bnds(lp, constraintID, GLP_UP, 0., normalizedOutputVal);    // <= outputPrec_i
                    ++constraintID;
                }

                // -(f(x_i) - t_i) <= outputPrec_i * xi
                // => f(x_i) + outputPrec_i * xi >= t_i
                {
                    for (unsigned int j = 0; j < nbTerms; ++j)
                    {
                        iMat[coeffID] = constraintID;
                        jMat[coeffID] = firstWeightParamID + j;
                        cMat[coeffID] = terms[j];                                           // weight_j * f_j
                        ++coeffID;
                    }

                    iMat[coeffID] = constraintID;
                    jMat[coeffID] = firstXiParamID + i;
                    cMat[coeffID] = normalizedOutputPrec;                                   // outputPrec_i * xi_i
                    ++coeffID;

                    glp_set_row_bnds(lp, constraintID, GLP_LO, normalizedOutputVal, 0.);    // >= outputPrec_i
                    ++constraintID;
                }
            }

            // On the derivatives of the polynomial
            {
                auto const termsDerivatives = getTermsDerivatives(normalizedInput, degree, isTermActive);
                for (unsigned int k = 0; k < nbUsedInputDims; ++k)
                {
                    double const inputPrec_k = normalizedInputPrecisions[k];
                    // Note: outputPrec_i is there because beta_ik represents a part of xi,
                    // which is normalized with regard to the output precision

                    // f'_k(x_i) * inputPrec_ik <= outputPrec_i * beta_ik
                    // => f'_k(x_i) * inputPrec_ik - outputPrec_i * beta_ik <= 0
                    // where f(x_i) = sum_j(weight_j * f_j(x_i))
                    // and f_j the j-th polynomial term
                    // with '_k the notation for the partial derivativation with regards to the k-th dimension in x_i
                    {
                        vector<double> allTermsKDerivative(nbTerms, 0.);
                        for (auto const& termDer : termsDerivatives[k])
                        {
                            allTermsKDerivative[termDer.first] = termDer.second;
                        }

                        for (unsigned int j = 0; j < nbTerms; ++j)
                        {
                            iMat[coeffID] = constraintID;
                            jMat[coeffID] = firstWeightParamID + j;
                            cMat[coeffID] = allTermsKDerivative[j] * inputPrec_k;           // weight_j * f'_jk
                            ++coeffID;
                        }

                        iMat[coeffID] = constraintID;
                        jMat[coeffID] = firstBetaIParamID + k;
                        cMat[coeffID] = -normalizedOutputPrec;                              // -outputPrec_i * beta_ik
                        ++coeffID;

                        glp_set_row_bnds(lp, constraintID, GLP_UP, 0., 0.);                 // <= 0
                        ++constraintID;
                    }

                    // -f'_k(x_i) * inputPrec_ik <= outputPrec_i * beta_ik
                    // => -f'_k(x_i) * inputPrec_ik - outputPrec_i * beta_ik <= 0
                    {

                        vector<double> allTermsKDerivative(nbTerms, 0.);
                        for (auto const& termDer : termsDerivatives[k])
                        {
                            allTermsKDerivative[termDer.first] = termDer.second;
                        }

                        for (unsigned int j = 0; j < nbTerms; ++j)
                        {
                            iMat[coeffID] = constraintID;
                            jMat[coeffID] = firstWeightParamID + j;
                            cMat[coeffID] = -allTermsKDerivative[j] * inputPrec_k;          // -weight_j * f'_jk
                            ++coeffID;
                        }

                        iMat[coeffID] = constraintID;
                        jMat[coeffID] = firstBetaIParamID + k;
                        cMat[coeffID] = -normalizedOutputPrec;                              // -outputPrec_i * beta_ik
                        ++coeffID;

                        glp_set_row_bnds(lp, constraintID, GLP_UP, 0., 0.);                 // <= 0
                        ++constraintID;
                    }
                }
            }
        }
    }

    // Now we solve the optimization problem and deduce if the form fit the data
    glp_load_matrix(lp, maxNbNonZeroCoeffs, iMat.data(), jMat.data(), cMat.data());
    glp_smcp glpParams;
    glp_init_smcp(&glpParams);
    glpParams.msg_lev = GLP_MSG_OFF;
    glpParams.presolve = GLP_ON;


    // Now we solve the optimization problem and deduce if the form fit the data
    auto const status = glp_simplex(lp, &glpParams);
    if ((status != 0) && (status != GLP_ENOPFS))
    {
        glp_delete_prob(lp);
        return make_tuple(false, false, Fitness<Polynomial>{});         // (a problem occured, no solution found)
    }

    if (status == GLP_ENOPFS)
    {
        glp_delete_prob(lp);
        return make_tuple(true, false, Fitness<Polynomial>{});          // (no problem occured, no solution found which satisfies all constraints)
    }

    // We retrieve the solution
    vector<double> solution(firstWeightParamID + nbOptimParameters);
    for (unsigned int ID = firstWeightParamID; ID < (firstWeightParamID + nbOptimParameters); ++ID)
    {
        solution[ID] = glp_get_col_prim(lp, ID);
    }

    glp_delete_prob(lp);


    if (mode != FitMode::WeightConcentration)
    {
        // We need to check if we found a solution such that for all i
        // |f(x_i)-t_i| <= outputPrec_i + sum_k(|f'_k(x_i)| * intputPrec_ik)
        // Because of the constraints, it's enough to check that for all i
        // outputPrec_i * xi_i <= outputPrec_i + sum_k(|f'_k(x_i)| * intputPrec_ik)
        unsigned int i = 0;
        for (auto mIt = mBegin; mIt != mEnd; ++mIt, ++i)
        {
            // Normalized output precision
            auto const normalizedOutputPrec = mIt->t[outputID][dimInOutput].precision * outputNormalizationFactor;

            // Normalized input
            vector<double> normalizedInput;
            vector<double> normalizedInputPrecisions;
            normalizedInput.reserve(nbUsedInputDims);
            normalizedInputPrecisions.reserve(nbUsedInputDims);

            for (auto const& id : usedInputDimIds)
            {
                normalizedInput.push_back(mIt->x[id].value * inputNormalizationFactors[id]);
                normalizedInputPrecisions.push_back(mIt->x[id].precision * inputNormalizationFactors[id]);
            }

            // Let's compute the right hand side
            double rhs = normalizedOutputPrec;

            auto const termsDerivatives = getTermsDerivatives(normalizedInput, degree, isTermActive);
            for (unsigned int k = 0; k < nbUsedInputDims; ++k)
            {
                double derivativeWithRegardToDim = 0.;

                for (auto const& termDer : termsDerivatives[k])
                {
                    derivativeWithRegardToDim += solution[firstWeightParamID + termDer.first] * termDer.second;
                }

                rhs += abs(derivativeWithRegardToDim) * normalizedInputPrecisions[k];
            }

            // Now let's check that the inequality is verified
            if (normalizedOutputPrec * solution[firstXiParamID + i] > rhs)
            {
                return make_tuple(true, false, Fitness<Polynomial>{});  // (same as above)
            }
        }
    }

    form.params.resize(nbUsedInputDims+1);  // Removing params that may remain from a previous call

    for (unsigned int j = 0; j < nbTerms; ++j)
    {
        form.params.push_back(solution[firstWeightParamID + j]);
    }

    if (mode == FitMode::Initial)
    {
        unsigned int const nbBetaParams = nbPoints * nbUsedInputDims;
        for (unsigned int ik = 0; ik < nbBetaParams; ++ik)
        {
            form.params.push_back(solution[firstBetaParamID + ik]);
        }
    }

    return make_tuple(true, true, Fitness<Polynomial>{nbTerms});        // (no problem occured, solution(s) found)
}



// Same thing with soplex
tuple<bool,bool,Fitness<Polynomial>> Polynomial::tryFitSoplex(
        FormType& form, unsigned int nbPoints,
        Models::ModelConstIterator<Polynomial> const& mBegin,
        Models::ModelConstIterator<Polynomial> const& mEnd,
        unsigned int outputID, unsigned int dimInOutput, FitMode mode,
        vector<bool> const& isTermActive)
{
    auto const usedInputDimIds = form.usedDimensions.getIds();
    unsigned int const nbUsedInputDims = form.usedDimensions.getNbUsed();
    unsigned int const totalNbInputDims = form.usedDimensions.getTotalNbDimensions();
    unsigned int const degree = form.degree;
    unsigned int const nbTerms = getNbTerms(nbUsedInputDims, degree, isTermActive);

    // Extracting the normalization factors
    vector<double> inputNormalizationFactors(totalNbInputDims);

    auto pIt = form.params.begin();
    for (auto const& id : usedInputDimIds)
    {
        inputNormalizationFactors[id] = *pIt;
        ++pIt;
    }

    double const maxOutput = form.params[nbUsedInputDims];
    double outputNormalizationFactor = 0.;
    if (maxOutput > 0.)
    {
        outputNormalizationFactor = 1./maxOutput;
    }


    // Let's create the optimization problem
    //
    // Naming convention:
    //    - i the index over points
    //    - j the index over polynomial terms
    //    - k the index over dimensions
    //
    // Optimization parameters:
    //    - 1 param per term (weight_j)
    //    - 1 param per point (xi_i) for the normalized distance to the approximation
    //    - 1 param (alpha) for the shared part of the xi_i which is due to the imprecision
    //      on the output values
    //    - 1 param per xi_i (i.e. per point) and per input dimension (beta_ik)
    //      for the part of those xi_i which is due to the imprecision on the input values
    //    - In terms of priorities,
    //      we want to minimize all xi_i first, then all beta_ik
    // When mode is WeightMinimization:
    //    - 1 param per term (gamma_j) is added to bound the absolute values of the weights
    //    - the beta_ik are bounded by their values in the previous call
    //    - what we minimize is the sum of the gamma_j,
    //      using values from the previous call to weigh their importance
    soplex::SoPlex problem;
    problem.setIntParam(soplex::SoPlex::OBJSENSE, soplex::SoPlex::OBJSENSE_MINIMIZE);
    problem.setIntParam(soplex::SoPlex::VERBOSITY, soplex::SoPlex::VERBOSITY_ERROR);

    unsigned int const nbOptimParameters =
            nbTerms                                 // weight_i
            + nbPoints                              // xi_i
            + 1                                     // alpha
            + nbUsedInputDims*nbPoints              // beta_ik
            + ((mode == FitMode::WeightConcentration)?
                nbTerms : 0);                       // gamma_i

    // Optimization parameters declaration
    {
        // Coefficients in what we minimize
        double xiCoeff = 0.;
        double alphaCoeff = 0.5;
        double betaCoeff = 1.;

        if (mode == FitMode::WeightConcentration)
        {
            xiCoeff = 0.;
            alphaCoeff = 0.;
            betaCoeff = 0.;
        }


        soplex::DSVector dummycol(0);

        // weight_j (unbounded)
        for (unsigned int j = 0; j < nbTerms; ++j)
        {
            problem.addColReal(soplex::LPCol(0., dummycol, soplex::infinity, -soplex::infinity));
        }

        // xi_i (positive)
        for (unsigned int i = 0; i < nbPoints; ++i)
        {
            problem.addColReal(soplex::LPCol(xiCoeff, dummycol, soplex::infinity, 0.));
        }

        // alpha (in [0,1])
        problem.addColReal(soplex::LPCol(alphaCoeff, dummycol, 1., 0.));

        // beta_ik (positive)
        if (mode == FitMode::WeightConcentration)
        {
            // The betas are bounded by their values in the previous call
            unsigned int const nbBetas = nbUsedInputDims*nbPoints;
            unsigned int const firstBetaInFormParams = form.params.size() - nbBetas;
            for (unsigned int i = 0; i < nbPoints; ++i)
            {
                unsigned int firstBetaInFormParamsForThisI = firstBetaInFormParams + i*nbUsedInputDims;

                for (unsigned int k = 0; k < nbUsedInputDims; ++k)
                {
                    problem.addColReal(soplex::LPCol(betaCoeff, dummycol, 1.1*
                                                     form.params[firstBetaInFormParamsForThisI + k], 0.));
                }
            }
        }
        else
        {
            for (unsigned int i = 0; i < nbPoints; ++i)
                for (unsigned int k = 0; k < nbUsedInputDims; ++k)
                {
                    problem.addColReal(soplex::LPCol(betaCoeff, dummycol, soplex::infinity, 0.));
                }
        }

        // gamma_j (positive)
        if (mode == FitMode::WeightConcentration)
        {
            // We compute the sum of all the absolute values of the weights
            //  in the solution from the previous call
            unsigned int const firstWeightInFormParams = nbUsedInputDims + 1;
            vector<double> absoluteWeights;
            absoluteWeights.reserve(nbTerms);

            for (unsigned int j = 0; j < nbTerms; ++j)
            {
                absoluteWeights.push_back(abs(form.params[firstWeightInFormParams+j]));
            }

            double sumAbsWeights = 0.;
            for (auto const& w : absoluteWeights)
            {
                sumAbsWeights += w;
            }

            // The higher the weight a term had in the solution from the previous call,
            //  the lower the cost we place on it, so that weights are concentrated
            for (unsigned int j = 0; j < nbTerms; ++j)
            {
                double const gammaCoeff = sumAbsWeights - absoluteWeights[j];
                problem.addColReal(soplex::LPCol(gammaCoeff, dummycol, soplex::infinity, 0.));
            }
        }
    }

    unsigned int const firstWeightParamID = 0;
    unsigned int const firstXiParamID = firstWeightParamID + nbTerms;
    unsigned int const alphaParamID = firstXiParamID + nbPoints;
    unsigned int const firstBetaParamID = alphaParamID + 1;
    unsigned int const firstGammaParamID = firstBetaParamID + nbPoints*nbUsedInputDims;

    // Optimization constraints declaration
    {
        soplex::DSVector row(nbOptimParameters);

        if (mode == FitMode::WeightConcentration)
        {
            // Constraints over the weights
            for (unsigned int j = 0; j < nbTerms; ++j)
            {
                // w_j <= gamma_j
                // => w_j - gamma_j <= 0
                {
                    row.add(firstWeightParamID + j, 1.);                                // w_j
                    row.add(firstGammaParamID + j, -1.);                                // -gamma_j

                    problem.addRowReal(soplex::LPRow(-soplex::infinity, row, 0.));
                    row.clear();
                }

                // -w_j <= gamma_j
                // => -w_j - gamma_j <= 0
                {
                    row.add(firstWeightParamID + j, -1.);                               // -w_j
                    row.add(firstGammaParamID + j, -1.);                                // -gamma_j

                    problem.addRowReal(soplex::LPRow(-soplex::infinity, row, 0.));
                    row.clear();
                }
            }
        }

        // Constraints over each point
        unsigned int i = 0;
        for (auto mIt = mBegin; mIt != mEnd; ++mIt, ++i)
        {
            unsigned int firstBetaIParamID = firstBetaParamID + i*nbUsedInputDims;

            // Normalized values and precision for the output
            auto const normalizedOutputVal = mIt->t[outputID][dimInOutput].value * outputNormalizationFactor;
            auto const normalizedOutputPrec = mIt->t[outputID][dimInOutput].precision * outputNormalizationFactor;

            // Normalized input
            vector<double> normalizedInput, normalizedInputPrecisions;
            normalizedInput.reserve(nbUsedInputDims);
            normalizedInputPrecisions.reserve(nbUsedInputDims);

            for (auto const& id : usedInputDimIds)
            {
                normalizedInput.push_back(mIt->x[id].value * inputNormalizationFactors[id]);
                normalizedInputPrecisions.push_back(mIt->x[id].precision * inputNormalizationFactors[id]);
            }


            // On xi_i
            {
                // xi_i <= alpha + sum_k(beta_ik)
                // => - xi_i + alpha + sum_k(beta_ik) >= 0
                row.add(firstXiParamID + i, -1.);                                   // -xi_i
                row.add(alphaParamID,        1.);                                   // alpha

                for (unsigned int k = 0; k < nbUsedInputDims; ++k)
                {
                    row.add(firstBetaIParamID + k, 1.);                             // beta_ik
                }

                problem.addRowReal(soplex::LPRow(0., row, soplex::infinity));
                row.clear();
            }


            // On the values of the polynomial
            {
                vector<double> const terms = getTerms(normalizedInput, degree, isTermActive);

                // (f(x_i) - t_i) <= outputPrec_i * xi_i
                // => f(x_i) - outputPrec_i * xi_i <= t_i
                // where f(x_i) = sum_j(weight_j * f_j(x_i))
                // and f_j the j-th polynomial term
                {
                    for (unsigned int j = 0; j < nbTerms; ++j)
                    {
                        row.add(firstWeightParamID + j, terms[j]);              // weight_j * f_j
                    }

                    row.add(firstXiParamID + i, -normalizedOutputPrec);         // -outputPrec_i * xi_i

                    problem.addRowReal(soplex::LPRow(-soplex::infinity, row, normalizedOutputVal));
                    row.clear();
                }

                // -(f(x_i) - t_i) <= outputPrec_i * xi
                // => f(x_i) + outputPrec_i * xi >= t_i
                {
                    for (unsigned int j = 0; j < nbTerms; ++j)
                    {
                        row.add(firstWeightParamID + j, terms[j]);              // weight_j * f_j
                    }

                    row.add(firstXiParamID + i, normalizedOutputPrec);          // outputPrec_i * xi_i

                    problem.addRowReal(soplex::LPRow(normalizedOutputVal, row, soplex::infinity));
                    row.clear();
                }
            }

            // On the derivatives of the polynomial
            {
                auto const termsDerivatives = getTermsDerivatives(normalizedInput, degree, isTermActive);
                for (unsigned int k = 0; k < nbUsedInputDims; ++k)
                {
                    double const inputPrec_k = normalizedInputPrecisions[k];
                    // Note: outputPrec_i is there because beta_ik represents a part of xi,
                    // which is normalized with regard to the output precision

                    // f'_k(x_i) * inputPrec_ik <= outputPrec_i * beta_ik
                    // => f'_k(x_i) * inputPrec_ik - outputPrec_i * beta_ik <= 0
                    // where f(x_i) = sum_j(weight_j * f_j(x_i))
                    // and f_j the j-th polynomial term
                    // with '_k the notation for the partial derivativation with regards to the k-th dimension in x_i
                    {
                        for (auto const& termDer : termsDerivatives[k])
                        {
                            unsigned int const weightParamID = firstWeightParamID + termDer.first;
                            double const derivative = termDer.second;
                            row.add(weightParamID, derivative * inputPrec_k);   // weight_j * f'_jk
                        }

                        row.add(firstBetaIParamID + k, -normalizedOutputPrec);  // -outputPrec_i * beta_ik

                        problem.addRowReal(soplex::LPRow(-soplex::infinity, row, 0.));
                        row.clear();
                    }

                    // -f'_k(x_i) * inputPrec_ik <= outputPrec_i * beta_ik
                    // => -f'_k(x_i) * inputPrec_ik - outputPrec_i * beta_ik <= 0
                    {
                        for (auto const& termDer : termsDerivatives[k])
                        {
                            unsigned int const weightParamID = firstWeightParamID + termDer.first;
                            double const derivative = termDer.second;
                            row.add(weightParamID, -derivative * inputPrec_k);   // -weight_j * f'_jk
                        }

                        row.add(firstBetaIParamID + k, -normalizedOutputPrec);  // -outputPrec_i * beta_ik

                        problem.addRowReal(soplex::LPRow(-soplex::infinity, row, 0.));
                        row.clear();
                    }
                }
            }
        }
    }

    // Now we solve the optimization problem and deduce if the form fit the data
    auto const status = problem.solve();
    if ((status != soplex::SPxSolver::OPTIMAL)
        && (status != soplex::SPxSolver::INForUNBD))
    {
        return make_tuple(false, false, Fitness<Polynomial>{});         // (a problem occured, no solution found)
    }

    if (status == soplex::SPxSolver::INForUNBD)
    {
        return make_tuple(true, false, Fitness<Polynomial>{});          // (no problem occured, no solution found which satisfies all constraints)
    }

    // We retrieve the solution
    soplex::DVector primal(nbOptimParameters);
    problem.getPrimalReal(primal);
    vector<double> solution(firstWeightParamID + nbOptimParameters);
    for (unsigned int ID = firstWeightParamID; ID < (firstWeightParamID + nbOptimParameters); ++ID)
    {
        solution[ID] = primal[ID];
    }


    if (mode != FitMode::WeightConcentration)
    {
        // We need to check if we found a solution such that for all i
        // |f(x_i)-t_i| <= outputPrec_i + sum_k(|f'_k(x_i)| * intputPrec_ik)
        // Because of the constraints, it's enough to check that for all i
        // outputPrec_i * xi_i <= outputPrec_i + sum_k(|f'_k(x_i)| * intputPrec_ik)
        unsigned int i = 0;
        for (auto mIt = mBegin; mIt != mEnd; ++mIt, ++i)
        {
            // Normalized output precision
            auto const normalizedOutputPrec = mIt->t[outputID][dimInOutput].precision * outputNormalizationFactor;

            // Normalized input
            vector<double> normalizedInput;
            vector<double> normalizedInputPrecisions;
            normalizedInput.reserve(nbUsedInputDims);
            normalizedInputPrecisions.reserve(nbUsedInputDims);

            for (auto const& id : usedInputDimIds)
            {
                normalizedInput.push_back(mIt->x[id].value * inputNormalizationFactors[id]);
                normalizedInputPrecisions.push_back(mIt->x[id].precision * inputNormalizationFactors[id]);
            }

            // Let's compute the right hand side
            double rhs = normalizedOutputPrec;

            auto const termsDerivatives = getTermsDerivatives(normalizedInput, degree, isTermActive);
            for (unsigned int k = 0; k < nbUsedInputDims; ++k)
            {
                double derivativeWithRegardToDim = 0.;

                for (auto const& termDer : termsDerivatives[k])
                {
                    derivativeWithRegardToDim += solution[firstWeightParamID + termDer.first] * termDer.second;
                }

                rhs += abs(derivativeWithRegardToDim) * normalizedInputPrecisions[k];
            }

            // Now let's check that the inequality is verified
            auto lhs = normalizedOutputPrec * solution[firstXiParamID + i];
            if (lhs > rhs)
            {
                return make_tuple(true, false, Fitness<Polynomial>{});  // (same as above)
            }
        }
    }

    form.params.resize(nbUsedInputDims+1);  // Removing params that may remain from a previous call

    for (unsigned int j = 0; j < nbTerms; ++j)
    {
        form.params.push_back(solution[firstWeightParamID + j]);
    }

    if (mode == FitMode::Initial)
    {
        unsigned int const nbBetaParams = nbPoints * nbUsedInputDims;
        for (unsigned int ik = 0; ik < nbBetaParams; ++ik)
        {
            form.params.push_back(solution[firstBetaParamID + ik]);
        }
    }

    return make_tuple(true, true, Fitness<Polynomial>{nbTerms});        // (no problem occured, solution(s) found)
}



}}
