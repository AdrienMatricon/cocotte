
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
#include <glpk.h>
#include <soplex/src/soplex.h>
#include <cocotte/datatypes.h>
using Cocotte::DataPoint;
#include <cocotte/approximators/polynomial.h>



namespace Cocotte {
namespace Approximators {



// Returns the most complex forms under a certain complexity,
// for each combination of the formerly used dimensions and at most one other.
list <list<Form>> Polynomial::getMostComplexForms(UsedDimensions const& formerlyUsedDimensions, int maxComplexity)
{
    if (maxComplexity == 1)
    {
        // Only one form possible, a constant
        // (no dimension used, degree 0)
        Form form(UsedDimensions(formerlyUsedDimensions.getTotalNbDimensions(), list<int>()));
        form.complexity = 1;
        form.other = 0; // Degree
        return list <list<Form>>(1, list<Form>(1, form));
    }

    int const maxNbDims = formerlyUsedDimensions.getNbUsed() + 1;
    list <list<Form>> result;

    for (int i = 0; i < maxNbDims; ++i)
    {
        int const d = i+1;  // number of used dimensions
        int N = 1;  // degree
        int c = complexity(N, d);
        if (c > maxComplexity)
        {
            break;
        }

        // We search for the highest degree for d dimensions under the complexity
        while (c < maxComplexity)
        {
            c = complexity(++N, d);
        }

        if (c > maxComplexity)
        {
            c = complexity(--N, d);
        }

        // We add all compute all corresponding forms and store them in a list
        list<UsedDimensions> dimCombinations = formerlyUsedDimensions.getCombinationsFromUsedAndOne(d);
        list<Form> current;
        for (auto& comb : dimCombinations)
        {
            Form form(std::move(comb));
            form.complexity = c;
            form.other = N;
            current.push_back(std::move(form));
        }

        // We insert the new list of forms while keeping sure that
        // the list of lists is sorted by increasing complexity
        auto rIt = result.begin();
        auto const rEnd = result.end();

        for (; rIt != rEnd; ++rIt)
        {
            if (rIt->back().complexity > c)
            {
                result.insert(rIt, std::move(current));
                break;
            }
        }

        if (rIt == rEnd)
        {
            result.insert(rIt, current);
        }

    }

    return result;
}

// Tries to fit the points with a form
// If success, params are stored within the form
bool Polynomial::tryFitGLPK(Form& form, int nbPoints, Models::ModelConstIterator mBegin, Models::ModelConstIterator mEnd, int outputID, int dimInOutput)
{
    auto const usedInputDimIds = form.usedDimensions.getIds();
    int const nbUsedInputDims = form.usedDimensions.getNbUsed();
    int const totalNbInputDims = form.usedDimensions.getTotalNbDimensions();
    int const degree = form.other;
    double const quantum = 1.e-6;   // small value for double comparison and to avoid dividing by 0


    // We determine the maximum absolute value for each used dimension and the output
    vector<double> maxes(totalNbInputDims, quantum), normalizationFactors(totalNbInputDims);
    double maxOuput = quantum;

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
        if (val > maxOuput)
        {
            maxOuput = val;
        }
    }


    // We compute normalization coefficients and store them in form.params
    form.params.clear();
    form.params.reserve(nbUsedInputDims+1);

    for (auto const& id : usedInputDimIds)
    {
        normalizationFactors[id] = 1/maxes[id];
        form.params.push_back(normalizationFactors[id]);
    }
    form.params.push_back(maxOuput);


    // We create the optimization problem
    glp_prob *lp;
    lp = glp_create_prob();
    glp_set_obj_dir(lp, GLP_MIN);

    int const nbTerms = getNbTerms(nbUsedInputDims, degree);
    int const nbVars = nbTerms + 1;

    // We add variables
    glp_add_cols(lp, nbVars);

    // Positive slack variable equal to the cost function
    glp_set_col_bnds(lp, 1, GLP_LO, 0.0, 0.0);
    glp_set_obj_coef(lp, 1, 1.0);

    // Unbound parameters
    int const supI = nbVars + 1;
    for (int i = 2; i < supI; ++i)
    {
        glp_set_col_bnds(lp, i, GLP_FR, 0.0, 0.0);
        glp_set_obj_coef(lp, i, 0.0);
    }

    // We add constraints (2 per datapoint)
    int const nbConstraints = 2*nbPoints;
    glp_add_rows(lp, nbConstraints);
    int const nbCoeffs = nbVars * nbConstraints;
    int iMat[1+nbCoeffs];       // Coefficient constraint IDs
    int jMat[1+nbCoeffs];       // Coefficient variable IDs
    double cMat[1+nbCoeffs];    // Coefficient values
    int constraintId = 1;
    int coeffId = 1;

    for (auto mIt = mBegin; mIt != mEnd; ++mIt)
    {
        auto const normalizedOutputVal = mIt->t[outputID][dimInOutput].value/maxOuput;
        auto const normalizedOutputPrec = mIt->t[outputID][dimInOutput].precision/maxOuput;

        vector<double> normalizedInput;
        normalizedInput.reserve(nbUsedInputDims);

        for (auto const& id : usedInputDimIds)
        {
            normalizedInput.push_back(mIt->x[id].value * normalizationFactors[id]);
        }

        // Polynomial terms
        vector<double> const terms = getTerms(normalizedInput, nbUsedInputDims, degree);

        // f(x) + prec * slack >= t
        {
            int varId = 1;

            // Slack
            iMat[coeffId] = constraintId;
            jMat[coeffId] = varId;
            cMat[coeffId] = normalizedOutputPrec;
            ++varId; ++coeffId;

            for (int i = 0; i < nbTerms; ++i)
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

            for (int i = 0; i < nbTerms; ++i)
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

    glp_load_matrix(lp, nbCoeffs, iMat, jMat, cMat);
    glp_smcp glpParams;
    glp_init_smcp(&glpParams);
    glpParams.msg_lev = GLP_MSG_OFF;


    // Now we solve the optimization problem and deduce if the form fit the data
    if (glp_simplex(lp, &glpParams) != 0)
    {
        cerr << endl;
        cerr << "No solution despite slack variables. Should not happen, check your data" << endl;
        cerr << "(in particular, check that there is no negative precision)" << endl;
        glp_delete_prob(lp);
        exit(1);
    }

    if (glp_get_col_prim(lp, 1) > 1.0)
    {
        glp_delete_prob(lp);
        return false;   // The solution does not satisfy all constraints
    }

    for (int i = 2; i < supI; ++i)
    {
        form.params.push_back(glp_get_col_prim(lp, i));
    }

    glp_delete_prob(lp);

    return true;
}



bool Polynomial::tryFit(Form& form, int nbPoints, Models::ModelConstIterator mBegin, Models::ModelConstIterator mEnd, int outputID, int dimInOutput)
{
    auto const usedInputDimIds = form.usedDimensions.getIds();
    int const nbUsedInputDims = form.usedDimensions.getNbUsed();
    int const totalNbInputDims = form.usedDimensions.getTotalNbDimensions();
    int const degree = form.other;
    double const quantum = 1.e-6;   // small value for double comparison and to avoid dividing by 0


    // We determine the maximum absolute value for each used dimension and the output
    vector<double> maxes(totalNbInputDims, quantum), normalizationFactors(totalNbInputDims);
    double maxOuput = quantum;

    for (auto mIt = mBegin; mIt != mEnd; ++mIt)
    {
        auto glou = *mIt;

        for (auto const& id : usedInputDimIds)
        {
            double const val = abs(mIt->x[id].value);
            if (val > maxes[id])
            {
                maxes[id] = val;
            }
        }

        double const val = abs(mIt->t[outputID][dimInOutput].value);
        if (val > maxOuput)
        {
            maxOuput = val;
        }
    }


    // We compute normalization coefficients and store them in form.params
    form.params.clear();
    form.params.reserve(nbUsedInputDims+1);

    for (auto const& id : usedInputDimIds)
    {
        normalizationFactors[id] = 1/maxes[id];
        form.params.push_back(normalizationFactors[id]);
    }
    form.params.push_back(maxOuput);


    // We create the optimization problem
    soplex::SoPlex problem;
    problem.setIntParam(soplex::SoPlex::OBJSENSE, soplex::SoPlex::OBJSENSE_MINIMIZE);
    problem.setIntParam(soplex::SoPlex::VERBOSITY, soplex::SoPlex::VERBOSITY_ERROR);

    int const nbTerms = getNbTerms(nbUsedInputDims, degree);
    int const nbVars = nbTerms + 1;

    // We add variables
    soplex::DSVector dummycol(0);
    problem.addColReal(soplex::LPCol(1.0, dummycol, soplex::infinity, 0.0));    // positive slack variable equal to the cost function
    for (int i = 0; i < nbTerms; ++i)
    {
        problem.addColReal(soplex::LPCol(0., dummycol, soplex::infinity, -soplex::infinity));  // unbound parameters
    }

    // We add constraints (2 per datapoint)
    soplex::DSVector row(nbVars);

    for (auto mIt = mBegin; mIt != mEnd; ++mIt)
    {
        // Normalized values and precision for the output
        auto const normalizedOutputVal = mIt->t[outputID][dimInOutput].value/maxOuput;
        auto const normalizedOutputPrec = mIt->t[outputID][dimInOutput].precision/maxOuput;

        // Normalized input
        vector<double> normalizedInput;
        normalizedInput.reserve(nbUsedInputDims);

        for (auto const& id : usedInputDimIds)
        {
            normalizedInput.push_back(mIt->x[id].value * normalizationFactors[id]);
        }

        // Polynomial terms
        vector<double> const terms = getTerms(normalizedInput, nbUsedInputDims, degree);

        // f(x) + prec * slack >= t
        {
            row.add(0, normalizedOutputPrec);    // slack

            for (int i = 0; i < nbTerms; ++i)
            {
               row.add(i+1, terms[i]);
            }

            problem.addRowReal(soplex::LPRow(normalizedOutputVal, row, soplex::infinity));
            row.clear();
        }

        // f(x) - prec * slack <= t
        {
            row.add(0, -normalizedOutputPrec);   // slack

            for (int i = 0; i < nbTerms; ++i)
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
        // Okay, I know this is ugly, but both glpk and soplex tend to randomly
        // return that they faced numerical instabilities instead of returning
        // the solution (which exists and is bounded), and it just so happens
        // that they don't seem to fail on the same problems...
        // I'm interested if anyone knows a usable library for linear program solving
        // that does not induces these sorts of problems
        // (especially if the library is efficient when called repeatedly and/or
        // is under a license that is less constraining than GPL or the ZIB academic license)

        return tryFitGLPK(form, nbPoints, mBegin, mEnd, outputID, dimInOutput);
    }

    soplex::DVector primal(nbVars);
    problem.getPrimalReal(primal);

    if (primal[0] > 1.0)
    {
        return false;   // The solution does not satisfy all constraints
    }

    for (int i = 1; i < nbVars; ++i)
    {
        form.params.push_back(primal[i]);
    }

    return true;
}


Form Polynomial::fitOnePoint(double t, int nbDims)
{
    Form form (UsedDimensions(nbDims, list<int>(), 0));
    form.complexity = 1;
    form.other = 0;
    form.params = vector<double>{t,1};

    return form;
}


// Estimates the value for the given inputs
vector<double> Polynomial::estimate(Form const&form, vector<vector<double>> const& points)
{
    int const nbPoints = points.size();
    int const nbDims = form.usedDimensions.getNbUsed();
    int const degree = form.other;

    auto const pBegin = form.params.begin();

    vector<double> result;
    result.reserve(nbPoints);

    vector<vector<double>> processedPoints;

    // Selecting dimensions and normalizing
    {
        processedPoints.reserve(nbPoints);
        list<int> const& dimIds = form.usedDimensions.getIds();
        auto const dBegin = dimIds.begin(), dEnd = dimIds.end();

        for (int i = 0; i < nbPoints; ++i)
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
        vector<double> const terms = Polynomial::getTerms(point, nbDims, degree);
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
string Polynomial::formToString(Form const& form, vector<string> inputNames)
{
    int const nbDims = form.usedDimensions.getNbUsed();
    int const degree = form.other;

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
    result += ", d = {";

    if (nbDims > 0)
    {
        result += dimNames[0];
    }

    for (int i = 1; i < nbDims; ++i)
    {
        result += "," + dimNames[i];
    }

    result += "}";

    return result;
}


// Evaluates all terms of the polynomial and returns them as a vector
vector<double> Polynomial::getTerms(vector<double> const& vals, int nbDims, int degree)
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
    int const nbTermsForOneDim = degree + 1;
    vector<list<double>> terms;
    terms.reserve(nbTermsForOneDim);

    auto const vBegin = vals.begin(), vEnd = vals.end();

    // We initialize terms by adding the first dimension terms to it
    {
        double const val = *vBegin;
        double temp = val;

        terms.push_back(list<double>(1, 1.));
        terms.push_back(list<double>(1, temp));

        for (int N = 2; N < nbTermsForOneDim; ++N)
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

            int const endLoop = degree;
            for (int M = 1; M < endLoop; ++M)
            {
                // M is the degree in the old dimensions
                int const newDeg = M + 1;
                for (auto& term : terms[M])
                {
                    newTerms[newDeg].push_back(term * temp);
                }
            }
        }

        // We add all other terms
        for (int N = 2; N < nbTermsForOneDim; ++N)
        {
            // N is the degree in the new dimension
            temp *= val;
            newTerms[N].push_back(temp);

            int const endLoop = degree - N + 1;
            for (int M = 1; M < endLoop; ++M)
            {
                // M is the degree in the old dimensions
                int const newDeg = M + N;
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
int Polynomial::getNbTerms(int nbDims, int degree)
{
    // nbDims among degree + nbDims
    int numerator = 1;
    int denominator = 1;
    int const sum = nbDims + degree;

    for (int i = 0; i < nbDims; ++i)
    {
        numerator *= sum - i;
        denominator *= nbDims - i;
    }

    auto const result = numerator / denominator;

    return result;
}



// Complexity definition
int Polynomial::complexity(int degree, int nbUsedDimensions)
{
    int const nbApproximablePoints = (degree * nbUsedDimensions) + 1;
    return nbApproximablePoints;
}



}}
