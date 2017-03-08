
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
#include <algorithm>
using std::min;
using std::max;
#include <tuple>
using std::tuple;
using std::make_tuple;
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
        unsigned int N = 1;     // degree
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
// using exactly nbNewDimensions dimensions not in formerlyUsedDimensions
// Those forms are returned as a lists of sublists of forms, such that:
//   - forms in the same sublist have the same complexity
//   - sublists are sorted by (non strictly) increasing complexity
list<list<FormType>> Polynomial::getFormsInComplexityRange_implementation(
        UsedDimensions const& formerlyUsedDimensions,
        unsigned int nbNewDimensions,
        unsigned int minComplexity,
        unsigned int maxComplexity)
{
    // Trivial case
    if (minComplexity > maxComplexity)
    {
        return {};
    }

    unsigned int const nbAvailableDimensions = formerlyUsedDimensions.getNbUsed() + nbNewDimensions;

    if (formerlyUsedDimensions.getTotalNbDimensions() < nbAvailableDimensions)
    {
        return {};
    }

    unsigned int const maxNbDims = min(maxComplexity - 1u,  // c = N*d+1, thus d <= c-1 for N > 0
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
                    = formerlyUsedDimensions.getCombinationsWithKUnused(d, nbNewDimensions);

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
// If it succeeds, params are stored within the form
// The function returns whether is was a success
//   and (only if it succeeded) the fitness of the form that was found
//   (the lower the better)
tuple<bool,double> Polynomial::tryFit_implementation(
        FormType& form, unsigned int nbPoints,
        Models::ModelConstIterator<Polynomial> mBegin,
        Models::ModelConstIterator<Polynomial> mEnd,
        unsigned int outputID, unsigned int dimInOutput)
{
    // We compute normalization coefficients and store them in form.params
    // We also compute the amplitude of the output and store it in form.params
    {
        auto const usedInputDimIds = form.usedDimensions.getIds();
        unsigned int const nbUsedInputDims = form.usedDimensions.getNbUsed();
        unsigned int const totalNbInputDims = form.usedDimensions.getTotalNbDimensions();

        // We determine the maximum absolute value for each used dimension and the output
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
    }

    // We try to fit the form to the data
    auto const soplexResult = tryFitSoplex(form, mBegin, mEnd, outputID, dimInOutput);

    if (std::get<0>(soplexResult))
    {
        auto const glpkResult = tryFitGLPK(form, nbPoints, mBegin, mEnd, outputID, dimInOutput);

        if (std::get<0>(glpkResult))
        {
            cerr << "Neither soplex nor GLPK can solve this" << endl;
            exit(1);
        }
        else
        {
            return make_tuple(std::get<1>(glpkResult), std::get<2>(glpkResult));
        }
    }

    return make_tuple(std::get<1>(soplexResult), std::get<2>(soplexResult));
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
        vector<double> const terms = Polynomial::getTerms(point, degree);
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
string Polynomial::formToString_implementation(FormType const& form, vector<string> inputNames)
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
vector<double> Polynomial::getTerms(vector<double> const& vals, unsigned int degree)
{
    // We handle trivial cases first
    if (degree < 2)
    {
        if (degree < 1)
        {
            // If the degree is zero, we return a vector only containing a 1
            return vector<double>(1,1);
        }

        vector<double> result(1,1);
        result.insert(result.end(), vals.begin(), vals.end());
        return result;
    }

    // We create a vector terms such that
    // terms[i] contains all terms of degree i
    unsigned int const nbTermsForOneDim = degree + 1;
    vector<list<double>> terms;
    terms.reserve(nbTermsForOneDim);

    auto const vBegin = vals.begin(), vEnd = vals.end();

    // We initialize terms by adding the first dimension terms to it
    {
        double const val = *vBegin;
        double temp = val;

        terms.push_back(list<double>(1, 1.));
        terms.push_back(list<double>(1, temp));

        for (unsigned int N = 2; N < nbTermsForOneDim; ++N)
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

            unsigned int const endLoop = degree;
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
        for (unsigned int N = 2; N < nbTermsForOneDim; ++N)
        {
            // N is the degree in the new dimension
            temp *= val;
            newTerms[N].push_back(temp);

            unsigned int const endLoop = degree - N + 1;
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

    return result;
}


// Number of terms returned by getTerms
unsigned int Polynomial::getNbTerms(unsigned int nbDims, unsigned int degree)
{
    // nbDims among degree + nbDims
    unsigned int numerator = 1;
    unsigned int denominator = 1;
    unsigned int const sum = nbDims + degree;

    for (unsigned int i = 0; i < nbDims; ++i)
    {
        numerator *= sum - i;
        denominator *= nbDims - i;
    }

    auto const result = numerator / denominator;

    return result;
}



// Complexity definition
unsigned int Polynomial::complexity(unsigned int degree, unsigned int nbUsedDimensions)
{
    return (degree * nbUsedDimensions) + 1;
}



// Trying to fit the polynomial to the points with GLPK
// - the first bool is set to true if a problem occured
// - the other values in the tuple are the actual return values
tuple<bool,bool,double> Polynomial::tryFitGLPK(
        FormType& form,unsigned int nbPoints,
        Models::ModelConstIterator<Polynomial> mBegin,
        Models::ModelConstIterator<Polynomial> mEnd,
        unsigned int outputID, unsigned int dimInOutput)
{
    auto const usedInputDimIds = form.usedDimensions.getIds();
    unsigned int const nbUsedInputDims = form.usedDimensions.getNbUsed();
    unsigned int const totalNbInputDims = form.usedDimensions.getTotalNbDimensions();
    unsigned int const degree = form.degree;

    vector<double> inputNormalizationFactors(totalNbInputDims);

    auto pIt = form.params.begin();
    for (auto const& id : usedInputDimIds)
    {
        inputNormalizationFactors[id] = *pIt;
        ++pIt;
    }

    double const maxOutput = form.params.back();
    double outputNormalizationFactor = 0.;
    if (maxOutput > 0.)
    {
        outputNormalizationFactor = 1./maxOutput;
    }

    // We create the optimization problem
    glp_prob *lp;
    lp = glp_create_prob();
    glp_set_obj_dir(lp, GLP_MIN);

    unsigned int const nbTerms = getNbTerms(nbUsedInputDims, degree);
    unsigned int const nbVars = nbTerms + 1;

    // We add variables
    glp_add_cols(lp, nbVars);

    // Positive slack variable equal to the cost function
    glp_set_col_bnds(lp, 1, GLP_LO, 0.0, 0.0);
    glp_set_obj_coef(lp, 1, 1.0);

    // Unbound parameters
    unsigned int const supI = nbVars + 1;
    for (unsigned int i = 2; i < supI; ++i)
    {
        glp_set_col_bnds(lp, i, GLP_FR, 0.0, 0.0);
        glp_set_obj_coef(lp, i, 0.0);
    }

    // We add constraints (2 per datapoint)
    unsigned int const nbConstraints = 2*nbPoints;
    glp_add_rows(lp, nbConstraints);
    unsigned int const nbCoeffs = nbVars * nbConstraints;
    vector<int> iMat(1+nbCoeffs);       // Coefficient constraint IDs
    vector<int>  jMat(1+nbCoeffs);      // Coefficient variable IDs
    vector<double> cMat(1+nbCoeffs);    // Coefficient values
    int constraintId = 1;
    int coeffId = 1;

    for (auto mIt = mBegin; mIt != mEnd; ++mIt)
    {
        auto const normalizedOutputVal = mIt->t[outputID][dimInOutput].value * outputNormalizationFactor;
        auto const normalizedOutputPrec = mIt->t[outputID][dimInOutput].precision * outputNormalizationFactor;

        vector<double> normalizedInput;
        normalizedInput.reserve(nbUsedInputDims);

        for (auto const& id : usedInputDimIds)
        {
            normalizedInput.push_back(mIt->x[id].value * inputNormalizationFactors[id]);
        }

        // Polynomial terms
        vector<double> const terms = getTerms(normalizedInput, degree);

        // f(x) + prec * slack >= t
        {
            int varId = 1;

            // Slack
            iMat[coeffId] = constraintId;
            jMat[coeffId] = varId;
            cMat[coeffId] = normalizedOutputPrec;
            ++varId; ++coeffId;

            for (unsigned int i = 0; i < nbTerms; ++i)
            {
                iMat[coeffId] = constraintId;
                jMat[coeffId] = varId;
                cMat[coeffId] = terms[i];

                ++varId; ++coeffId;
            }
            glp_set_row_bnds(lp, constraintId, GLP_LO, normalizedOutputVal, normalizedOutputVal);
            ++constraintId;
        }

        // f(x) - prec * slack <= t
        {
            int varId = 1;

            // Slack
            iMat[coeffId] = constraintId;
            jMat[coeffId] = varId;
            cMat[coeffId] = -normalizedOutputPrec;
            ++varId; ++coeffId;

            for (unsigned int i = 0; i < nbTerms; ++i)
            {
                iMat[coeffId] = constraintId;
                jMat[coeffId] = varId;
                cMat[coeffId] = terms[i];
                ++varId; ++coeffId;
            }

            glp_set_row_bnds(lp, constraintId, GLP_UP, normalizedOutputVal, normalizedOutputVal);
            ++constraintId;
        }
    }

    glp_load_matrix(lp, nbCoeffs, iMat.data(), jMat.data(), cMat.data());
    glp_smcp glpParams;
    glp_init_smcp(&glpParams);
    glpParams.msg_lev = GLP_MSG_OFF;


    // Now we solve the optimization problem and deduce if the form fit the data
    if (glp_simplex(lp, &glpParams) != 0)
    {
        glp_delete_prob(lp);
        return make_tuple(true, false, 0.);                     // (a problem occured, no solution found)
    }

    if (glp_get_col_prim(lp, 1) > 1.0)
    {
        glp_delete_prob(lp);
        return make_tuple(false, false, 0.);                    // (no problem occured, no solution found which satisfies all constraints)
    }

    for (unsigned int i = 2; i < supI; ++i)
    {
        form.params.push_back(glp_get_col_prim(lp, i));
    }

    glp_delete_prob(lp);

    return make_tuple(false, true, glp_get_col_prim(lp, 1));    // (no problem occured, solution(s) found)
}



// Same thing with soplex
tuple<bool,bool,double> Polynomial::tryFitSoplex(
        FormType& form,
        Models::ModelConstIterator<Polynomial> mBegin,
        Models::ModelConstIterator<Polynomial> mEnd,
        unsigned int outputID, unsigned int dimInOutput)
{
    auto const usedInputDimIds = form.usedDimensions.getIds();
    unsigned int const nbUsedInputDims = form.usedDimensions.getNbUsed();
    unsigned int const totalNbInputDims = form.usedDimensions.getTotalNbDimensions();
    unsigned int const degree = form.degree;

    vector<double> inputNormalizationFactors(totalNbInputDims);

    auto pIt = form.params.begin();
    for (auto const& id : usedInputDimIds)
    {
        inputNormalizationFactors[id] = *pIt;
        ++pIt;
    }

    double const maxOutput = form.params.back();
    double outputNormalizationFactor = 0.;
    if (maxOutput > 0.)
    {
        outputNormalizationFactor = 1./maxOutput;
    }


    // We create the optimization problem
    soplex::SoPlex problem;
    problem.setIntParam(soplex::SoPlex::OBJSENSE, soplex::SoPlex::OBJSENSE_MINIMIZE);
    problem.setIntParam(soplex::SoPlex::VERBOSITY, soplex::SoPlex::VERBOSITY_ERROR);

    unsigned int const nbTerms = getNbTerms(nbUsedInputDims, degree);
    unsigned int const nbVars = nbTerms + 1;

    // We add variables
    soplex::DSVector dummycol(0);
    problem.addColReal(soplex::LPCol(1.0, dummycol, soplex::infinity, 0.0));    // positive slack variable equal to the cost function
    for (unsigned int i = 0; i < nbTerms; ++i)
    {
        problem.addColReal(soplex::LPCol(0., dummycol, soplex::infinity, -soplex::infinity));  // unbound parameters
    }

    // We add constraints (2 per datapoint)
    soplex::DSVector row(nbVars);

    for (auto mIt = mBegin; mIt != mEnd; ++mIt)
    {
        // Normalized values and precision for the output
        auto const normalizedOutputVal = mIt->t[outputID][dimInOutput].value * outputNormalizationFactor;
        auto const normalizedOutputPrec = mIt->t[outputID][dimInOutput].precision * outputNormalizationFactor;

        // Normalized input
        vector<double> normalizedInput;
        normalizedInput.reserve(nbUsedInputDims);

        for (auto const& id : usedInputDimIds)
        {
            normalizedInput.push_back(mIt->x[id].value * inputNormalizationFactors[id]);
        }

        // Polynomial terms
        vector<double> const terms = getTerms(normalizedInput, degree);

        // f(x) + prec * slack >= t
        {
            row.add(0, normalizedOutputPrec);    // slack

            for (unsigned int i = 0; i < nbTerms; ++i)
            {
                row.add(i+1, terms[i]);
            }

            problem.addRowReal(soplex::LPRow(normalizedOutputVal, row, soplex::infinity));
            row.clear();
        }

        // f(x) - prec * slack <= t
        {
            row.add(0, -normalizedOutputPrec);   // slack

            for (unsigned int i = 0; i < nbTerms; ++i)
            {
                row.add(i+1, terms[i]);
            }

            problem.addRowReal(soplex::LPRow(-soplex::infinity, row, normalizedOutputVal));
            row.clear();
        }
    }


    // Now we solve the optimization problem and deduce if the form fit the data
    if (problem.solve() != soplex::SPxSolver::OPTIMAL)
    {
        return make_tuple(true, false, 0.);     // (a problem occured, no solution found)
    }

    soplex::DVector primal(nbVars);
    problem.getPrimalReal(primal);

    if (primal[0] > 1.0)
    {
        return make_tuple(false, false, 0.);    // (no problem occured, no solution found which satisfies all constraints)
    }

    for (unsigned int i = 1; i < nbVars; ++i)
    {
        form.params.push_back(primal[i]);
    }

    return make_tuple(false, true, primal[0]);  // (no problem occured, solution(s) found)
}



}}
