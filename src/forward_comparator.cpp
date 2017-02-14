
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
#include <cmath>
using std::abs;
using std::sqrt;
using std::pow;
#include <algorithm>
#include <vector>
using std::vector;
#include <utility>
using std::pair;
#include <unordered_map>
using std::unordered_map;
#include <string>
using std::string;
#include <sstream>
using std::stringstream;
#include <fstream>
using std::ofstream;
#include <random>

#include <functionapproximators/FunctionApproximatorGMR.hpp>
#include <functionapproximators/FunctionApproximatorIRFRLS.hpp>
#include <functionapproximators/FunctionApproximatorLWPR.hpp>
#include <functionapproximators/FunctionApproximatorLWR.hpp>
#include <functionapproximators/FunctionApproximatorRBFN.hpp>
#include <functionapproximators/FunctionApproximatorGPR.hpp>
#include <functionapproximators/MetaParametersGMR.hpp>
#include <functionapproximators/MetaParametersIRFRLS.hpp>
#include <functionapproximators/MetaParametersLWPR.hpp>
#include <functionapproximators/MetaParametersLWR.hpp>
#include <functionapproximators/MetaParametersRBFN.hpp>
#include <functionapproximators/MetaParametersGPR.hpp>

#include <datasources/datasource.h>
using DataSources::TestData;
#include <datasources/dataloader.h>
using DataSources::DataLoader;
#include <cocotte/datatypes.h>
using Cocotte::Measure;
using Cocotte::DataPoint;
#include <cocotte/learner.h>
using Cocotte::Learner;
#include <cocotte/approximators/polynomial.h>
using Cocotte::Approximators::Polynomial;



void displayUsage();

void parseArguments(int argc, char *argv[],
                    string& trainingDataFile, string& testDataFile, string& validationDataFile,
                    string& trainingDataStructureFile, string& testDataStructureFile, string& validationDataStructureFile,
                    string& outputFile, string& cocotteDumpFile,
                    bool& defaultBatchSize, unsigned int& batchSize,
                    unsigned int& firstBatchID, bool& defaultLastBatchID, unsigned int& lastBatchID,
                    unordered_map<string, unsigned int> const& methodIDs, vector<bool>& isMethodUsed);

double distanceSquaredL2(vector<double> lhs, vector<Measure> rhs);
double distanceLinfinity(vector<double> lhs, vector<Measure> rhs);

vector<vector<vector<double>>> nearestNeighbourL2(vector<DataPoint> const& trainingDataPoints, TestData const& allTestDataPoints);
vector<vector<vector<double>>> nearestNeighbourLinfinity(vector<DataPoint> const& trainingDataPoints, TestData const& allTestDataPoints);

vector<unsigned int> countWrongPredictions(vector<vector<vector<double>>> const& estimations,
                                           vector<vector<vector<double>>> const& targets,
                                           vector<vector<vector<double>>> const& targetsPrec);

vector<double> meanPredictionL2Error(vector<vector<vector<double>>> const& estimations,
                                     vector<vector<vector<double>>> const& targets);

vector<double> meanPredictionL2NormalizedError(vector<vector<vector<double>>> const& estimations,
                                               vector<vector<vector<double>>> const& targets,
                                               vector<vector<vector<double>>> const& targetsPrec);

double meanPredictionMatrixL2NormalizedErrorOneOutput(
        Eigen::MatrixXd const& estimations,
        vector<vector<vector<double>>> const& targets,
        vector<vector<vector<double>>> const& targetsPrec,
        unsigned int outputID);

void addLine(string const& outputFileName, unsigned int methodID,
             unsigned int nbTrainingPoints, unsigned int nbTestPoints,
             vector<unsigned int> const& nbWrongPredictions,
             vector<double> const& meanL2Error, vector<double> const& meanL2NormalizedError);




int main(int argc, char *argv[])
{
    // Learning methods
    unordered_map<string, unsigned int> const methodIDs(
    { {"NN_L2", 0},
      {"NN_Linf", 1},
      {"RBFN", 2},
      {"LWR", 3},
      {"LWPR", 4},
      {"iRFRLS", 5},
      {"GPR", 6},
      {"GMR3", 7},
      {"GMR30", 8},
      {"GMR300", 9},
      {"GMRopti", 10},
      {"cocotte", 11} }
                );

    unsigned int const nbMethods = methodIDs.size();

    // Parameters
    string trainingDataFile, testDataFile, validationDataFile,
            trainingDataStructureFile, testDataStructureFile, validationDataStructureFile,
            outputFile = "forward_comparator_output.csv", cocotteDumpFile = "";
    bool defaultBatchSize = true, defaultLastBatchID = true;
    unsigned int batchSize, firstBatchID = 0, lastBatchID;
    vector<bool> isMethodUsed(10, false);

    // We parse the arguments
    {
        parseArguments(argc, argv,
                       trainingDataFile, testDataFile, validationDataFile,
                       trainingDataStructureFile, testDataStructureFile, validationDataStructureFile,
                       outputFile, cocotteDumpFile,
                       defaultBatchSize, batchSize,
                       firstBatchID, defaultLastBatchID, lastBatchID,
                       methodIDs, isMethodUsed);

        if (!defaultLastBatchID && (firstBatchID > lastBatchID))
        {
            cerr << "The IDs of the batches with which computation begins and ends have incompatible values" << endl;
            return 1;
        }
    }

    vector<DataPoint> allTrainingDataPoints;
    vector<string> inputVariableNames;
    vector<vector<string>> outputVariableNames;
    TestData allTestDataPoints, allValidationDataPoints;

    // Reading training data
    {
        DataLoader source(trainingDataFile, trainingDataStructureFile);
        unsigned int const nbTrainingPoints = source.getNbDataPoints();

        if (defaultBatchSize)
        {
            if (defaultLastBatchID)
            {
                lastBatchID = firstBatchID;
            }

            batchSize = nbTrainingPoints / (lastBatchID + 1);
        }
        else if (defaultLastBatchID)
        {
            lastBatchID = (nbTrainingPoints / batchSize) - 1;
        }
        else if (nbTrainingPoints <= (batchSize * lastBatchID))
        {
            cerr << "Not enough training datapoints to reach the last batch" << endl;
            cerr << "(" << nbTrainingPoints << " vs " << (batchSize * lastBatchID) + 1 << "needed at minimum)" << endl;
            return 1;
        }

        allTrainingDataPoints = source.getTrainingDataPoints(nbTrainingPoints);
        inputVariableNames = source.getInputVariableNames();
        outputVariableNames = source.getOutputVariableNames();
    }

    // Reading test data
    {
        DataLoader source(testDataFile, testDataStructureFile, inputVariableNames, outputVariableNames);

        unsigned int const nbTestDataPoints = source.getNbDataPoints();
        if (nbTestDataPoints == 0)
        {
            cerr << "No test datapoints" << endl;
            return 1;
        }

        allTestDataPoints = source.getTestDataPoints(nbTestDataPoints);
    }

    // Reading validation data
    {
        DataLoader source(validationDataFile, validationDataStructureFile, inputVariableNames, outputVariableNames);

        unsigned int const nbValidationDataPoints = source.getNbDataPoints();
        if (nbValidationDataPoints == 0)
        {
            cerr << "No validation datatpoints" << endl;
            return 1;
        }

        allValidationDataPoints = source.getTestDataPoints(nbValidationDataPoints);
    }

    cout << "Data successfully loaded." << endl;


    // Expressing some constants and declaring some variables
    unsigned int const nbInputDims = inputVariableNames.size();
    unsigned int const nbOutputs = outputVariableNames.size();
    unsigned int const nbTestPoints = allTestDataPoints.xValues.size();
    unsigned int const nbValidationPoints = allValidationDataPoints.xValues.size();

    Eigen::MatrixXd testInputMatrix, validationInputMatrix;

    if (isMethodUsed[methodIDs.at("RBFN")]
            || isMethodUsed[methodIDs.at("LWR")]
            || isMethodUsed[methodIDs.at("LWPR")]
            || isMethodUsed[methodIDs.at("iRFRLS")]
            || isMethodUsed[methodIDs.at("GPR")]
            || isMethodUsed[methodIDs.at("GMR3")]
            || isMethodUsed[methodIDs.at("GMR30")]
            || isMethodUsed[methodIDs.at("GMR300")]
            || isMethodUsed[methodIDs.at("GMRopti")])
    {
        testInputMatrix = Eigen::MatrixXd(nbTestPoints, nbInputDims);

        for (unsigned int i = 0; i < nbTestPoints; ++i)
            for (unsigned int j = 0; j < nbInputDims; ++j)
            {
                testInputMatrix(i,j) = allTestDataPoints.xValues[i][j];
            }
    }

    if (isMethodUsed[methodIDs.at("LWPR")]
            || isMethodUsed[methodIDs.at("GMRopti")])
    {
        validationInputMatrix = Eigen::MatrixXd(nbValidationPoints, nbInputDims);

        for (unsigned int i = 0; i < nbValidationPoints; ++i)
            for (unsigned int j = 0; j < nbInputDims; ++j)
            {
                validationInputMatrix(i,j) = allValidationDataPoints.xValues[i][j];
            }
    }


    // Processing the batches
    for (unsigned int batchID = firstBatchID; batchID <= lastBatchID; ++batchID)
    {
        vector<DataPoint> trainingDataPoints;

        // Getting the training DataPoints from all batches up to this one
        {
            unsigned int nbTrainingPoints = batchSize * (batchID + 1);
            if (nbTrainingPoints > allTrainingDataPoints.size())
            {
                nbTrainingPoints = allTrainingDataPoints.size();
            }

            trainingDataPoints.insert(trainingDataPoints.end(), allTrainingDataPoints.begin(), allTrainingDataPoints.begin() + nbTrainingPoints);

            cout << endl << "Batch " << batchID << " (" << nbTrainingPoints << " training points in total):" << endl;
        }

        Eigen::MatrixXd trainingInputMatrix;
        vector<Eigen::MatrixXd> trainingOutputMatrix;
        unsigned int const nbTrainingPoints = trainingDataPoints.size();

        if (isMethodUsed[methodIDs.at("RBFN")]
                || isMethodUsed[methodIDs.at("LWR")]
                || isMethodUsed[methodIDs.at("LWPR")]
                || isMethodUsed[methodIDs.at("iRFRLS")]
                || isMethodUsed[methodIDs.at("GPR")]
                || isMethodUsed[methodIDs.at("GMR3")]
                || isMethodUsed[methodIDs.at("GMR30")]
                || isMethodUsed[methodIDs.at("GMR300")]
                || isMethodUsed[methodIDs.at("GMRopti")])
        {
            trainingInputMatrix = Eigen::MatrixXd(nbTrainingPoints, nbInputDims);

            for (unsigned int i = 0; i < nbTrainingPoints; ++i)
                for (unsigned int j = 0; j < nbInputDims; ++j)
                {
                    trainingInputMatrix(i,j) = trainingDataPoints[i].x[j].value;
                }

            trainingOutputMatrix.reserve(nbOutputs);

            for (unsigned int k = 0; k < nbOutputs; ++k)
            {
                unsigned int const nbDims = outputVariableNames[k].size();
                Eigen::MatrixXd out(nbTrainingPoints, nbDims);
                for (unsigned int i = 0; i < nbTrainingPoints; ++i)
                    for (unsigned int j = 0; j < nbDims; ++j)
                    {
                        out(i,j) = trainingDataPoints[i].t[k][j].value;
                    }
                trainingOutputMatrix.push_back(out);
            }
        }


        // Evaluating the various methods

        // Nearest Neighbour, using L2 distance
        for (unsigned int methodID = 0; methodID < nbMethods; ++methodID)
        {
            if (!isMethodUsed[methodID])
            {
                continue;
            }

            vector<vector<vector<double>>> predictions;
            vector<Eigen::MatrixXd> predictionMatrices(nbOutputs);

            if (methodID == methodIDs.at("NN_L2"))
            {
                cout << "  NN_L2:" << endl;

                cout << "    Making predictions..."; cout.flush();
                predictions = nearestNeighbourL2(trainingDataPoints, allTestDataPoints);
                cout << " done." << endl;
            }

            else if (methodID == methodIDs.at("NN_Linf"))
            {
                cout << "  NN_Linf:" << endl;

                cout << "    Making predictions..."; cout.flush();
                predictions = nearestNeighbourLinfinity(trainingDataPoints, allTestDataPoints);
                cout << " done." << endl;
            }

            else if (methodID == methodIDs.at("RBFN"))
            {
                cout << "  RBFN:" << endl;

                cout << "    Training and making predictions..."; cout.flush();

                Eigen::VectorXi nbCentersPerDim(nbInputDims);
                unsigned int const roughNbCenters = nbTrainingPoints/10;
                unsigned int const perDim = 1u + static_cast<unsigned int>(pow(roughNbCenters,1./nbInputDims));
                for (unsigned int i = 0; i < nbInputDims; ++i)
                {
                    nbCentersPerDim(i) = perDim;
                }

                DmpBbo::MetaParametersRBFN const metaParameters(nbInputDims, nbCentersPerDim);

                for (unsigned int k = 0; k < nbOutputs; ++k)
                {
                    DmpBbo::FunctionApproximatorRBFN rbfn(&metaParameters);
                    rbfn.train(trainingInputMatrix, trainingOutputMatrix[k]);
                    rbfn.predict(testInputMatrix, predictionMatrices[k]);
                }
                cout << " done." << endl;
            }

            else if (methodID == methodIDs.at("LWR"))
            {
                cout << "  LWR:" << endl;

                cout << "    Training and making predictions..."; cout.flush();

                // The LWPR implementation fails and predicts NaN if there are too many centers
                Eigen::VectorXi nbCentersPerDim(nbInputDims);
                unsigned int const roughNbCenters = nbTrainingPoints/10;
                unsigned int const perDim = 1u + static_cast<unsigned int>(pow(roughNbCenters,1./nbInputDims));
                for (unsigned int i = 0; i < nbInputDims; ++i)
                {
                    nbCentersPerDim(i) = perDim;
                }

                DmpBbo::MetaParametersLWR const metaParameters(nbInputDims, nbCentersPerDim);

                for (unsigned int k = 0; k < nbOutputs; ++k)
                {
                    DmpBbo::FunctionApproximatorLWR lwr(&metaParameters);
                    lwr.train(trainingInputMatrix, trainingOutputMatrix[k]);
                    lwr.predict(testInputMatrix, predictionMatrices[k]);
                }
                cout << " done." << endl;
            }

            else if (methodID == methodIDs.at("LWPR"))
            {
                cout << "  LWPR:" << endl;

                cout << "    Training and making predictions..."; cout.flush();

                Eigen::MatrixXd temp;
                std::default_random_engine generator;
                std::uniform_real_distribution<double> distribution(0.0,1.0);

                for (unsigned int k = 0; k < nbOutputs; ++k)
                {
                    double error, newError, referenceError;
                    // init_D, w_gen, w_prune
                    vector<double> params{0.05, 0.1, 0.9};
                    vector<double> paramVariation{0.1, 0.5, 0.5};
                    vector<double> newParams = params, referenceParams;
                    unsigned int currentParam = 0;
                    bool update_D = false;
                    unsigned int optimPhase = 0;

                    while (optimPhase < 7)
                    {
                        cout << "param " << currentParam << ": " << params[currentParam] << " vs " << newParams[currentParam] << endl;
                        DmpBbo::MetaParametersLWPR const metaParameters(
                                    nbInputDims, newParams[0]*Eigen::VectorXd::Ones(1), newParams[1], newParams[2], update_D);
                        DmpBbo::FunctionApproximatorLWPR lwpr(&metaParameters);
                        lwpr.train(trainingInputMatrix, trainingOutputMatrix[k]);
                        lwpr.predict(validationInputMatrix, temp);
                        newError = meanPredictionMatrixL2NormalizedErrorOneOutput(
                                    temp, allValidationDataPoints.tValues, allValidationDataPoints.tPrecisions, k);

                        cout << "=> err: " << error << " vs " << newError << endl << endl;

                        if (optimPhase == 0)
                        {
                            error = newError;
                            referenceError = newError;
                            referenceParams = newParams;

                            if ((currentParam == 1) || (currentParam == 2))
                            {
                                newParams[currentParam] =
                                        (9. * newParams[currentParam] + paramVariation[currentParam])/10.;
                            }
                            else
                            {
                                newParams[currentParam] += paramVariation[currentParam];
                            }
                            ++optimPhase;
                            continue;
                        }
                        else if ((optimPhase % 2) == 1)
                        {
                            if (newError < 10 &&
                                    ((newError < 0.0001)
                                     || (abs(newError - error)/(newError + error) < 0.005)))
                            {
                                params = newParams;
                                optimPhase += 2;

                                if (optimPhase == 7)
                                {
                                    update_D = true;
                                }
                                else
                                {
                                    ++currentParam;
                                    if ((currentParam == 1) || (currentParam == 2))
                                    {
                                        newParams[currentParam] =
                                                (9. * newParams[currentParam] + paramVariation[currentParam])/10.;
                                    }
                                    else
                                    {
                                        newParams[currentParam] += paramVariation[currentParam];
                                    }
                                }

                                continue;
                            }
                            else if (newError > 10 || newError < error)
                            {
                                params = newParams;
                                error = newError;

                                if ((currentParam == 1) || (currentParam == 2))
                                {
                                    newParams[currentParam] =
                                            (9. * newParams[currentParam] + paramVariation[currentParam])/10.;
                                }
                                else
                                {
                                    newParams[currentParam] += paramVariation[currentParam];
                                }
                                continue;
                            }
                            else
                            {
                                referenceError = newError;
                                referenceParams = newParams;

                                double ratio = distribution(generator);
                                newParams[currentParam] = ratio*params[currentParam] + (1.-ratio) * referenceParams[currentParam];

                                ++optimPhase;
                                continue;
                            }
                        }
                        else if ((optimPhase % 2) == 0)
                        {
                            if ((newError < 0.01)
                                    || (abs(newError - error)/(newError + error) < 0.005)
                                    || (abs(newParams[currentParam] - params[currentParam]) < 0.01))
                            {
                                params = newParams;
                                ++optimPhase;

                                if (optimPhase == 7)
                                {
                                    update_D = true;
                                }
                                else
                                {
                                    ++currentParam;
                                    newParams[currentParam] += paramVariation[currentParam];
                                }

                                continue;
                            }
                            else if (newError < error)
                            {
                                params = newParams;
                                error = newError;

                                double ratio = distribution(generator);
                                newParams[currentParam] = ratio*params[currentParam] + (1.-ratio) * referenceParams[currentParam];

                                continue;
                            }
                            else if (newError < referenceError)
                            {
                                referenceError = newError;
                                referenceParams = newParams;

                                double ratio = distribution(generator);
                                newParams[currentParam] = ratio*params[currentParam] + (1.-ratio) * referenceParams[currentParam];

                                continue;
                            }
                            else
                            {
                                double ratio = distribution(generator);
                                newParams[currentParam] = ratio*params[currentParam] + (1.-ratio) * referenceParams[currentParam];
                                continue;
                            }
                        }
                    }

                    DmpBbo::MetaParametersLWPR const metaParameters(
                                nbInputDims, params[0]*Eigen::VectorXd::Ones(1), params[1], params[2], update_D);
                    DmpBbo::FunctionApproximatorLWPR lwpr(&metaParameters);
                    lwpr.train(trainingInputMatrix, trainingOutputMatrix[k]);
                    lwpr.predict(testInputMatrix, predictionMatrices[k]);
                }
                cout << " done." << endl;
            }

            else if (methodID == methodIDs.at("iRFRLS"))
            {
                cout << "  iRFRLS:" << endl;

                cout << "    Training and making predictions..."; cout.flush();
                int const nbBasisFunctions = 100;
                double const lambda = 0.2;
                double const gamma = 10.;
                DmpBbo::MetaParametersIRFRLS const metaParameters(nbInputDims, nbBasisFunctions, lambda, gamma);

                for (unsigned int k = 0; k < nbOutputs; ++k)
                {
                    DmpBbo::FunctionApproximatorIRFRLS iRfrls(&metaParameters);
                    iRfrls.train(trainingInputMatrix, trainingOutputMatrix[k]);
                    iRfrls.predict(testInputMatrix, predictionMatrices[k]);
                }
                cout << " done." << endl;
            }

            else if (methodID == methodIDs.at("GPR"))
            {
                cout << "  GPR:" << endl;

                cout << "    Training and making predictions..."; cout.flush();
                double const maxCovariance = 3.;
                double const length = 0.1;
                DmpBbo::MetaParametersGPR const metaParameters(nbInputDims, maxCovariance, length);

                for (unsigned int k = 0; k < nbOutputs; ++k)
                {
                    DmpBbo::FunctionApproximatorGPR gpr(&metaParameters);
                    gpr.train(trainingInputMatrix, trainingOutputMatrix[k]);
                    gpr.predict(testInputMatrix, predictionMatrices[k]);
                }
                cout << " done." << endl;
            }

            else if (methodID == methodIDs.at("GMR3"))
            {
                cout << "  GMR3:" << endl;

                cout << "    Training and making predictions..."; cout.flush();
                // We ensure 1 <= nbGaussians <= nbTrainingPoints/2 to avoid problems
                unsigned int const nbGaussians = std::min(3u, std::max(1u, nbTrainingPoints/2));
                DmpBbo::MetaParametersGMR const metaParameters(nbInputDims, nbGaussians);

                for (unsigned int k = 0; k < nbOutputs; ++k)
                {
                    DmpBbo::FunctionApproximatorGMR gmr(&metaParameters);
                    gmr.train(trainingInputMatrix, trainingOutputMatrix[k]);
                    gmr.predict(testInputMatrix, predictionMatrices[k]);
                }
                cout << " done." << endl;
            }

            else if (methodID == methodIDs.at("GMR30"))
            {
                cout << "  GMR30:" << endl;

                cout << "    Training and making predictions..."; cout.flush();
                // We ensure 1 <= nbGaussians <= nbTrainingPoints/2 to avoid problems
                unsigned int const nbGaussians = std::min(30u, std::max(1u, nbTrainingPoints/2));
                DmpBbo::MetaParametersGMR const metaParameters(nbInputDims, nbGaussians);

                for (unsigned int k = 0; k < nbOutputs; ++k)
                {
                    DmpBbo::FunctionApproximatorGMR gmr(&metaParameters);
                    gmr.train(trainingInputMatrix, trainingOutputMatrix[k]);
                    gmr.predict(testInputMatrix, predictionMatrices[k]);
                }
                cout << " done." << endl;
            }

            else if (methodID == methodIDs.at("GMR300"))
            {
                cout << "  GMR300:" << endl;

                cout << "    Training and making predictions..."; cout.flush();
                // We ensure 1 <= nbGaussians <= nbTrainingPoints/2 to avoid problems
                unsigned int const nbGaussians = std::min(300u, std::max(1u, nbTrainingPoints/2));
                DmpBbo::MetaParametersGMR const metaParameters(nbInputDims, nbGaussians);

                for (unsigned int k = 0; k < nbOutputs; ++k)
                {
                    DmpBbo::FunctionApproximatorGMR gmr(&metaParameters);
                    gmr.train(trainingInputMatrix, trainingOutputMatrix[k]);
                    gmr.predict(testInputMatrix, predictionMatrices[k]);
                }
                cout << " done." << endl;
            }

            else if (methodID == methodIDs.at("GMRopti"))
            {
                cout << "  GMRopti:" << endl;

                cout << "    Training and making predictions..."; cout.flush();
                for (unsigned int k = 0; k < nbOutputs; ++k)
                {
                    unsigned int minNbGaussians = 1, maxNbGaussians = std::min(300u, std::max(1u, nbTrainingPoints/2));
                    unsigned int currentNbGaussians;
                    Eigen::MatrixXd temp;
                    double minError, maxError;

                    // Min
                    {
                        DmpBbo::MetaParametersGMR const metaParameters(nbInputDims, minNbGaussians);
                        DmpBbo::FunctionApproximatorGMR gmr(&metaParameters);
                        gmr.train(trainingInputMatrix, trainingOutputMatrix[k]);
                        gmr.predict(validationInputMatrix, temp);
                        minError = meanPredictionMatrixL2NormalizedErrorOneOutput(
                                    temp, allValidationDataPoints.tValues, allValidationDataPoints.tPrecisions, k);
                    }

                    // Max
                    {
                        DmpBbo::MetaParametersGMR const metaParameters(nbInputDims, maxNbGaussians);
                        DmpBbo::FunctionApproximatorGMR gmr(&metaParameters);
                        gmr.train(trainingInputMatrix, trainingOutputMatrix[k]);
                        gmr.predict(validationInputMatrix, temp);
                        maxError = meanPredictionMatrixL2NormalizedErrorOneOutput(
                                    temp, allValidationDataPoints.tValues, allValidationDataPoints.tPrecisions, k);
                    }

                    while ( abs(minError - maxError)/(minError + maxError) > 0.005)
                    {
                        bool const replaceMin = minError > maxError;
                        currentNbGaussians = (minNbGaussians + maxNbGaussians) / 2;

                        if ( replaceMin && ((minNbGaussians + maxNbGaussians) % 2 != 0) )
                        {
                            ++currentNbGaussians;
                        }

                        DmpBbo::MetaParametersGMR const metaParameters(nbInputDims, currentNbGaussians);
                        DmpBbo::FunctionApproximatorGMR gmr(&metaParameters);
                        gmr.train(trainingInputMatrix, trainingOutputMatrix[k]);
                        gmr.predict(validationInputMatrix, temp);
                        double const currentError = meanPredictionMatrixL2NormalizedErrorOneOutput(
                                    temp, allValidationDataPoints.tValues, allValidationDataPoints.tPrecisions,  k);

                        if (currentError < 0.0001)
                        {
                            break;
                        }

                        if (replaceMin)
                        {
                            minNbGaussians = currentNbGaussians;
                            minError = currentError;
                        }
                        else
                        {
                            maxNbGaussians = currentNbGaussians;
                            maxError = currentError;
                        }
                    }

                    DmpBbo::MetaParametersGMR const metaParameters(nbInputDims, currentNbGaussians);
                    DmpBbo::FunctionApproximatorGMR gmr(&metaParameters);
                    gmr.train(trainingInputMatrix, trainingOutputMatrix[k]);
                    gmr.predict(testInputMatrix, predictionMatrices[k]);
                }
                cout << " done." << endl;
            }

            else if (methodID == methodIDs.at("cocotte"))
            {
                cout << "  cocotte:" << endl;

                cout << "    Training..."; cout.flush();
                Learner<Polynomial> learner(inputVariableNames, outputVariableNames);
                learner.addDataPointsIncremental(trainingDataPoints);
                cout << " done." << endl;

                if (cocotteDumpFile != "")
                {
                    cout << "    Dumping models..."; cout.flush();
                    learner.dumpModels(cocotteDumpFile);
                    cout << " done." << endl;
                }

                cout << "    Making predictions..."; cout.flush();
                predictions = learner.predict(allTestDataPoints.xValues);
                cout << " done." << endl;
            }

            // If necessary, we fill predictions with testPredictionMatrix
            if ( (methodID == methodIDs.at("RBFN"))
                 || (methodID == methodIDs.at("LWR"))
                 || (methodID == methodIDs.at("LWPR"))
                 || (methodID == methodIDs.at("iRFRLS"))
                 || (methodID == methodIDs.at("GPR"))
                 || (methodID == methodIDs.at("GMR3"))
                 || (methodID == methodIDs.at("GMR30"))
                 || (methodID == methodIDs.at("GMR300"))
                 || (methodID == methodIDs.at("GMRopti")) )
            {
                // Resizing predictions
                {
                    vector<vector<double>> temp;
                    temp.reserve(nbOutputs);
                    for (auto const& out : outputVariableNames)
                    {
                        temp.push_back(vector<double>(out.size()));
                    }
                    predictions.resize(nbTestPoints, temp);
                }

                // Filling predictions
                for (unsigned int i = 0; i < nbTestPoints; ++i)
                    for (unsigned int k = 0; k < nbOutputs; ++k)
                    {
                        unsigned int const nbDims = outputVariableNames[k].size();
                        for (unsigned int j = 0; j < nbDims; ++j)
                        {
                            predictions[i][k][j] = predictionMatrices[k](i,j);
                        }
                    }
            }

            cout << "    Evaluating the method..."; cout.flush();
            auto const nbWrongPredictions = countWrongPredictions(predictions, allTestDataPoints.tValues, allTestDataPoints.tPrecisions);
            auto const meanL2Error = meanPredictionL2Error(predictions, allTestDataPoints.tValues);
            auto const meanL2NormalizedError = meanPredictionL2NormalizedError(predictions, allTestDataPoints.tValues, allTestDataPoints.tPrecisions);
            addLine(outputFile, methodID, trainingDataPoints.size(), nbTestPoints, nbWrongPredictions, meanL2Error, meanL2NormalizedError);
            cout << " done." << endl;

        }
    }


    return 0;
}


void displayUsage()
{
    cout << "Usage: forward_comparator [OPTION]... TRAINING_DATA_FILE... TRAINING_DATA_STRUCTURE_FILE... TEST_DATA_FILE... [TEST_DATA_STRUCTURE_FILE]" << endl;
    cout << "Options:" << endl;
    cout << "-o, --output <file>" << "\t  " << "output file" << endl
         << "\t\t\t  " << "(default: 'forward_comparator_output.csv')" << endl;
    cout << "-v, --validation <file>" << "\t  " << "validation data file" << endl
         << "\t\t\t  " << "(default: training data file)" << endl;
    cout << "-V, --validation-structure <file>" << "\t  " << "validation data structure file" << endl
         << "\t\t\t  " << "(default: training data structure file)" << endl;
    cout << "-f, --first-batch-id <ID>"<< "  " << "ID of the batch with which computation begins:" << endl
         << "\t\t\t  " << "training starts with (firstBatchID+1) x batchSize points" << endl
         << "\t\t\t  " << "(default: 0)" << endl;
    cout << "-l, --last-batch-id <ID>"<< "  " << "ID of the batch with which computation ends:" << endl
         << "\t\t\t  " << "training ends with (lastBatchID+1) x batchSize points" << endl
         << "\t\t\t  " << "(default: the batch with the highest ID," << endl
         << "\t\t\t  "<< "or firstBatchID if the batch size was not specified)" << endl;
    cout << "-b, --batch-size <size>"<< "  " << "size of the successive batches of training datapoints" << endl
         << "\t\t\t  "<< "(default: nbTrainingDatapoints/(lastBatchID+1))" << endl;
    cout << "-m, --method <name>"<< "\t  " << "activates a method for which training is performed" << endl
         << "\t\t\t  " << "Can be: 'NN_L2, 'RBFN', 'LWR', 'LWPR', 'iRFRLS'," << endl
         << "\t\t\t  " << "'GPR', 'GMR3', 'GMR30', 'GMR300', 'GMRopti', 'cocotte', 'cocotte_classifier_Linf'" << endl
         << "\t\t\t  " << "or 'cocotte_classifier_relevant_Linf'" << endl
         << "\t\t\t  " << "Can also be 'all' to activate all methods" << endl
         << "\t\t\t  " << "(default: 'all')" << endl;
    cout << "    --c-dump <file>" << "\t  " << "if it is active, models learned by" << endl
         << "\t\t\t  " << "method 'cocotte' will be dumped in <file>" << endl;
}


void parseArguments(int argc, char *argv[],
                    string& trainingDataFile, string& testDataFile, string& validationDataFile,
                    string& trainingDataStructureFile, string& testDataStructureFile, string& validationDataStructureFile,
                    string& outputFile, string& cocotteDumpFile,
                    bool& defaultBatchSize, unsigned int& batchSize,
                    unsigned int& firstBatchID, bool& defaultLastBatchID, unsigned int& lastBatchID,
                    unordered_map<string, unsigned int> const& methodIDs, vector<bool>& isMethodUsed)
{
    bool correctSyntax = true, methodOptionUsed = false;
    unsigned int nbArguments = argc-1;

    for (unsigned int i = 1; i <= nbArguments; ++i)
    {
        auto current = argv[i];

        // Not an option
        if (current[0] != '-')
        {
            if (trainingDataFile.empty())
            {
                trainingDataFile = current;
            }
            else if (trainingDataStructureFile.empty())
            {
                trainingDataStructureFile = current;
            }
            else if (testDataFile.empty())
            {
                testDataFile = current;
            }
            else if (testDataStructureFile.empty())
            {
                testDataStructureFile = current;
            }
            else
            {
                cerr << "Too many arguments" << endl;
                correctSyntax = false;
            }
        }
        else    // options
        {
            if ((current == string("-h")) || (current == string("--help")))
            {
                displayUsage();
                exit(0);
            }
            else if ((current == string("-o")) || (current == string("--output")))
            {
                ++i;
                if (i <= nbArguments)
                {
                    outputFile = argv[i];
                    continue;
                }

                cerr << "Wrong use of option --output (-o)" << endl;
            }
            else if ((current == string("-v")) || (current == string("--validation")))
            {
                ++i;
                if (i <= nbArguments)
                {
                    validationDataFile = argv[i];
                    continue;
                }

                cerr << "Wrong use of option --validation (-v)" << endl;
            }
            else if ((current == string("-V")) || (current == string("--validation-structure")))
            {
                ++i;
                if (i <= nbArguments)
                {
                    validationDataStructureFile = argv[i];
                    continue;
                }

                cerr << "Wrong use of option --validation-structure (-V)" << endl;
            }
            else if ((current == string("-f")) || (current == string("--first-batch-id")))
            {
                ++i;
                if (i <= nbArguments)
                {
                    stringstream ss;
                    ss << argv[i];
                    ss >> firstBatchID;

                    if (!ss.fail())
                    {
                        continue;
                    }
                }

                cerr << "Wrong use of option --first-batch-id (-f)" << endl;
            }
            else if ((current == string("-l")) || (current == string("--last-batch-id")))
            {
                ++i;
                if (i <= nbArguments)
                {
                    stringstream ss;
                    ss << argv[i];
                    ss >> lastBatchID;
                    defaultLastBatchID = false;

                    if (!ss.fail())
                    {
                        continue;
                    }
                }

                cerr << "Wrong use of option --last-batch-id (-l)" << endl;
            }
            else if ((current == string("-b")) || (current == string("--batch-size")))
            {
                defaultBatchSize = false;

                ++i;
                if (i <= nbArguments)
                {
                    stringstream ss;
                    ss << argv[i];
                    ss >> batchSize;

                    if (!ss.fail())
                    {
                        continue;
                    }
                }

                cerr << "Wrong use of option --batch-size (-b)" << endl;
            }
            else if ((current == string("-m")) || (current == string("--method")))
            {
                methodOptionUsed = true;
                ++i;

                if (i <= nbArguments)
                {
                    string name = argv[i];

                    if (name == "all")
                    {
                        for (auto const& item : methodIDs)
                        {
                            if (isMethodUsed[item.second])
                            {
                                cerr << "Warning: method " << name << " activated more than once through option --method (-m)" << endl;
                            }
                            else
                            {
                                isMethodUsed[item.second] = true;
                            }
                        }
                        continue;
                    }
                    else
                    {
                        if (methodIDs.count(name) > 0)
                        {
                            auto const methodID = methodIDs.at(name);
                            if (isMethodUsed[methodID])
                            {
                                cerr << "Warning: method " << name << " activated more than once through option --method (-m)" << endl;
                            }
                            else
                            {
                                isMethodUsed[methodID] = true;
                            }

                            continue;
                        }

                        cerr << "Unknow method name " << name << " passed through option --method (-m)" << endl;
                    }
                }
            }
            else if (current == string("--c-dump"))
            {
                ++i;
                if (i <= nbArguments)
                {
                    cocotteDumpFile = argv[i];
                    continue;
                }

                cerr << "Wrong use of option --c-dump" << endl;
            }
            else
            {
                cerr << "Unkown option " << current << endl;
            }

            correctSyntax = false;
        }
    }

    // Missing arguments
    if (trainingDataFile.empty())
    {
        cerr << "Missing training data file name" << endl;
        correctSyntax = false;
    }

    if (trainingDataStructureFile.empty())
    {
        cerr << "Training data structure file name" << endl;
        correctSyntax = false;
    }

    if (testDataFile.empty())
    {
        cerr << "Missing test data file name" << endl;
        correctSyntax = false;
    }

    // Exiting if wrong syntax
    if (!correctSyntax)
    {
        cerr << "Try 'forward-comparator --help'' for more information." << endl;
        exit(1);
    }

    // Default values
    if (testDataStructureFile.empty())
    {
        testDataStructureFile = trainingDataStructureFile;
    }

    if (validationDataFile.empty())
    {
        validationDataFile = trainingDataFile;
    }

    if (validationDataStructureFile.empty())
    {
        validationDataStructureFile = trainingDataStructureFile;
    }

    if (!methodOptionUsed)
    {
        for (auto const& item : methodIDs)
        {
            isMethodUsed[item.second] = true;
        }
    }
}


double distanceSquaredL2(vector<double> lhs, vector<Measure> rhs)
{
    double distance = 0.;

    auto rIt = rhs.begin();
    auto const lEnd = lhs.end();
    for (auto lIt = lhs.begin(); lIt != lEnd; ++lIt, ++rIt)
    {
        auto temp = *lIt - rIt->value;
        distance += temp * temp;
    }

    return distance;
}


double distanceLinfinity(vector<double> lhs, vector<Measure> rhs)
{
    double distance = 0.;

    auto rIt = rhs.begin();
    auto const lEnd = lhs.end();
    for (auto lIt = lhs.begin(); lIt != lEnd; ++lIt, ++rIt)
    {
        auto temp = abs(*lIt - rIt->value);
        distance = std::max(distance, temp);
    }

    return distance;
}


vector<vector<vector<double>>> nearestNeighbourL2(vector<DataPoint> const& trainingDataPoints, TestData const& allTestDataPoints)
{
    vector<vector<vector<double>>> predictions;
    predictions.reserve(allTestDataPoints.xValues.size());

    vector<vector<double>> predictionFormat(trainingDataPoints[0].t.size());
    for (unsigned int i = 0; i < trainingDataPoints[0].t.size(); ++i)
    {
        predictionFormat[i].resize(trainingDataPoints[0].t[i].size());
    }

    for (auto const& x : allTestDataPoints.xValues)
    {
        auto bestPoint = trainingDataPoints[0];
        auto bestDistance = distanceSquaredL2(x, bestPoint.x);
        // We search for the closest point
        for (auto const& neighbour : trainingDataPoints)
        {
            auto const distance = distanceSquaredL2(x, neighbour.x);
            if (distance < bestDistance)
            {
                bestPoint = neighbour;
                bestDistance = distance;
            }
        }

        vector<vector<double>> tValues = predictionFormat;
        for (unsigned int i = 0; i < tValues.size(); ++i)
        {
            for (unsigned int j = 0; j < tValues[i].size(); ++j)
            {
                tValues[i][j] = bestPoint.t[i][j].value;
            }
        }
        predictions.push_back(tValues);
    }

    return predictions;
}


vector<vector<vector<double>>> nearestNeighbourLinfinity(vector<DataPoint> const& trainingDataPoints, TestData const& allTestDataPoints)
{
    vector<vector<vector<double>>> predictions;
    predictions.reserve(allTestDataPoints.xValues.size());

    vector<vector<double>> predictionFormat(trainingDataPoints[0].t.size());
    for (unsigned int i = 0; i < trainingDataPoints[0].t.size(); ++i)
    {
        predictionFormat[i].resize(trainingDataPoints[0].t[i].size());
    }

    for (auto const& x : allTestDataPoints.xValues)
    {
        auto bestPoint = trainingDataPoints[0];
        auto bestDistance = distanceLinfinity(x, bestPoint.x);
        // We search for the closest point
        for (auto const& neighbour : trainingDataPoints)
        {
            auto const distance = distanceLinfinity(x, neighbour.x);
            if (distance < bestDistance)
            {
                bestPoint = neighbour;
                bestDistance = distance;
            }
        }

        vector<vector<double>> tValues = predictionFormat;
        for (unsigned int i = 0; i < tValues.size(); ++i)
        {
            for (unsigned int j = 0; j < tValues[i].size(); ++j)
            {
                tValues[i][j] = bestPoint.t[i][j].value;
            }
        }
        predictions.push_back(tValues);
    }

    return predictions;
}


vector<unsigned int> countWrongPredictions(vector<vector<vector<double>>> const& estimations,
                                           vector<vector<vector<double>>> const& targets,
                                           vector<vector<vector<double>>> const& targetsPrec)
{
    unsigned int const nbPoints = estimations.size();
    unsigned int const nbOutputs = estimations.front().size();

    vector<unsigned int> result(nbOutputs, 0u);

    for (unsigned int k = 0; k < nbPoints; ++k)
        for (unsigned int i = 0; i < nbOutputs; ++i)
        {
            unsigned int const nbDims = targets[k][i].size();
            for (unsigned int j = 0; j < nbDims; ++j)
            {
                if (abs(estimations[k][i][j] - targets[k][i][j]) > targetsPrec[k][i][j])
                {
                    result[i] += 1;
                    break;
                }
            }
        }

    return result;
}

vector<double> meanPredictionL2Error(vector<vector<vector<double>>> const& estimations,
                                     vector<vector<vector<double>>> const& targets)
{
    unsigned int const nbPoints = estimations.size();
    unsigned int const nbOutputs = estimations.front().size();

    vector<double> result(nbOutputs, 0.);

    for (unsigned int k = 0; k < nbPoints; ++k)
        for (unsigned int i = 0; i < nbOutputs; ++i)
        {
            unsigned int const nbDims = targets[k][i].size();
            double error = 0.;
            for (unsigned int j = 0; j < nbDims; ++j)
            {
                double temp = estimations[k][i][j] - targets[k][i][j];
                error += temp * temp;
            }
            result[i] += sqrt(error);
        }

    for (double& res : result)
    {
        res /= nbPoints;
    }

    return result;
}

vector<double> meanPredictionL2NormalizedError(vector<vector<vector<double>>> const& estimations,
                                               vector<vector<vector<double>>> const& targets,
                                               vector<vector<vector<double>>> const& targetsPrec)
{
    unsigned int const nbPoints = estimations.size();
    unsigned int const nbOutputs = estimations.front().size();

    vector<double> result(nbOutputs, 0.);

    for (unsigned int k = 0; k < nbPoints; ++k)
        for (unsigned int i = 0; i < nbOutputs; ++i)
        {
            unsigned int const nbDims = targets[k][i].size();
            double error = 0.;
            for (unsigned int j = 0; j < nbDims; ++j)
            {
                double temp = (estimations[k][i][j] - targets[k][i][j])/targetsPrec[k][i][j];
                error += temp * temp;
            }
            result[i] += sqrt(error);
        }

    for (double& res : result)
    {
        res /= nbPoints;
    }

    return result;
}

double meanPredictionMatrixL2NormalizedErrorOneOutput(
        Eigen::MatrixXd const& estimations,
        vector<vector<vector<double>>> const& targets,
        vector<vector<vector<double>>> const& targetsPrec,
        unsigned int outputID)
{
    unsigned int const nbPoints = targets.size();

    double result = 0.;

    for (unsigned int i = 0; i < nbPoints; ++i)
    {
        unsigned int const nbDims = targets[i].size();
        double error = 0.;
        for (unsigned int j = 0; j < nbDims; ++j)
        {
            double temp = (estimations(i,j) - targets[i][outputID][j])/targetsPrec[i][outputID][j];
            error += temp * temp;
        }
        result += sqrt(error);
    }

    result /= nbPoints;

    return result;
}


void addLine(string const& outputFileName, unsigned int methodID,
             unsigned int nbTrainingPoints, unsigned int nbTestPoints,
             vector<unsigned int> const& nbWrongPredictions,
             vector<double> const& meanL2Error, vector<double> const& meanL2NormalizedError)
{
    ofstream outputFile(outputFileName, ofstream::app);
    if (!outputFile.is_open())
    {
        cerr << "Failed to open output file" << endl;
        exit(1);
    }

    // If the file was just created, we add the headers line
    if (outputFile.tellp() == 0)
    {
        outputFile << "method_ID, nb_training_points, nb_test_points";
        unsigned int const nbDims = nbWrongPredictions.size();
        for (unsigned int i = 0; i < nbDims; ++i)
        {
            outputFile << ", nb_wrong_predictions_" << i;
            outputFile << ", mean_prediction_L2_error_" << i;
            outputFile << ", mean_prediction_L2_normalized_error_" << i;
        }

        outputFile << endl;
    }

    // Then we write the data
    outputFile << methodID << ", "
               << nbTrainingPoints << ", "
               << nbTestPoints;

    {
        auto const nwpEnd = nbWrongPredictions.end();

        auto nwpIt = nbWrongPredictions.begin();
        auto mL2eIt = meanL2Error.begin();
        auto mL2neIt = meanL2NormalizedError.begin();

        for (; nwpIt != nwpEnd; ++nwpIt, ++mL2eIt, ++mL2neIt)
        {
            outputFile << ", " << *nwpIt;
            outputFile << ", " << *mL2eIt;
            outputFile << ", " << *mL2neIt;
        }
    }

    outputFile << endl;
}






