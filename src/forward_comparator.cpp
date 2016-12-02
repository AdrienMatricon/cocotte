
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
#include <cmath>
using std::abs;
#include <algorithm>
using std::max;
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
#include <cocotte/datatypes.h>
using Cocotte::Measure;
using Cocotte::DataPoint;
#include <datasources/datasource.h>
using DataSources::TestData;
#include <datasources/dataloader.h>
using DataSources::DataLoader;

void displayUsage();

bool parseArguments(int argc, char *argv[],
                    string& trainingDataFile, string& testDataFile,
                    string& trainingDataStructureFile, string& testDataStructureFile, string& outputFile,
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

void addLine(string outputFileName, unsigned int methodID,
             unsigned int nbTrainingPoints, unsigned int nbTestPoints,
             vector<unsigned int> nbWrongPredictions);




int main(int argc, char *argv[])
{
    // Methods
    unordered_map<string, unsigned int> const methodIDs(
    { {"NN_L2", 0},
      {"NN_Linf", 1},
      {"RBFN", 2},
      {"LWR", 3},
      {"LWPR", 4},
      {"iRFRLS", 5},
      {"GPR", 6},
      {"GMR", 7},
      {"cocotte", 8},
      {"cocotte_classifier_Linf", 9},
      {"cocotte_classifier_relevant_Linf", 10} }
                );

    // Parameters
    string trainingDataFile, testDataFile,
            trainingDataStructureFile, testDataStructureFile, outputFile = "forward_comparator_output.csv";
    bool defaultBatchSize = true, defaultLastBatchID = true;
    unsigned int batchSize, firstBatchID = 0, lastBatchID;
    vector<bool> isMethodUsed(10, false);

    // We parse the arguments
    {
        bool displayHelp = parseArguments(argc, argv,
                                          trainingDataFile, testDataFile,
                                          trainingDataStructureFile, testDataStructureFile, outputFile,
                                          defaultBatchSize, batchSize,
                                          firstBatchID, defaultLastBatchID, lastBatchID,
                                          methodIDs, isMethodUsed);

        if (displayHelp)
        {
            displayUsage();
            return 0;
        }

        if (!defaultLastBatchID && (firstBatchID > lastBatchID))
        {
            cerr << "The IDs of the batches with which computation begins and ends have incompatible values" << endl;
            return 1;
        }
    }

    vector<DataPoint> allTrainingDataPoints;
    vector<string> inputVariableNames;
    vector<vector<string>> outputVariableNames;
    TestData allTestDataPoints;

    // Reading training data
    {
        DataLoader source(trainingDataFile, trainingDataStructureFile);
        unsigned int const nbTrainingPoints = source.getNbDataPoints();

        if (defaultBatchSize)
        {
            if (defaultLastBatchID)
            {
                lastBatchID = 0;
            }

            batchSize = nbTrainingPoints / (lastBatchID + 1);
        }
        else if (defaultLastBatchID)
        {
            lastBatchID = (nbTrainingPoints / batchSize) - 1;
        }
        else if (nbTrainingPoints <= (batchSize * lastBatchID))
        {
            cerr << "Not enough datapoints to reach the last batch" << endl;
            return 1;
        }

        allTrainingDataPoints = source.getTrainingDataPoints(nbTrainingPoints);
        inputVariableNames = source.getInputVariableNames();
        outputVariableNames = source.getOutputVariableNames();
    }

    // Reading test data
    {
        DataLoader source(testDataFile, testDataStructureFile, inputVariableNames, outputVariableNames);
        allTestDataPoints = source.getTestDataPoints(source.getNbDataPoints());
    }

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

            trainingDataPoints.insert(trainingDataPoints.end(), trainingDataPoints.begin(), trainingDataPoints.begin() + nbTrainingPoints);
        }


        // Evaluating the various methods

        // Nearest Neighbour, using L2 distance
        for (unsigned int methodID = 0; methodID < 11; ++methodID)
        {
            if (!isMethodUsed[methodID])
            {
                continue;
            }

            vector<vector<vector<double>>> predictions;

            if (methodID == methodIDs.at("NN_L2"))
            {
                predictions = nearestNeighbourL2(trainingDataPoints, allTestDataPoints);
            }

            if (methodID == methodIDs.at("NN_Linf"))
            {
                predictions = nearestNeighbourLinfinity(trainingDataPoints, allTestDataPoints);
            }

            if (methodID == methodIDs.at("RBFN"))
            {

            }

            if (methodID == methodIDs.at("LWR"))
            {

            }

            if (methodID == methodIDs.at("LWPR"))
            {

            }

            if (methodID == methodIDs.at("iRFRLS"))
            {

            }

            if (methodID == methodIDs.at("GPR"))
            {

            }

            if (methodID == methodIDs.at("GMR"))
            {

            }

            if (methodID == methodIDs.at("cocotte"))
            {

            }

            if (methodID == methodIDs.at("cocotte_classifier_Linf"))
            {

            }

            if (methodID == methodIDs.at("cocotte_classifier_relevant_Linf"))
            {

            }

            auto const nbWrongPredictions = countWrongPredictions(predictions, allTestDataPoints.tValues, allTestDataPoints.tPrecisions);
            addLine(outputFile, methodID, trainingDataPoints.size(), allTestDataPoints.xValues.size(), nbWrongPredictions);
        }
    }


    return 0;
}


void displayUsage()
{
    cout << "Usage: forward_comparator [OPTION]... TRAINING_DATA_FILE... TRAINING_DATA_STRUCTURE_FILE... TEST_DATA_FILE... [TEST_DATA_STRUCTURE_FILE]" << endl;
    cout << "Options:" << endl;
    cout << "-o, --output" << "\t\t" << "output file (default: 'forward_comparator_output.csv')" << endl;
    cout << "-b, --batch-size"<< "\t" << "number of new training data points that will be added with each batch (default: all training datapoints)" << endl;
    cout << "-f, --first-batch-id"<< "\t" << "ID of the batch with which computation begins (default: 0)" << endl;
    cout << "-l, --last-batch-id"<< "\t" << "ID of the batch with which computation ends (default: the batch with the highest ID)" << endl;
    cout << "-m, --method"<< "\t\t" << "activates a method for which training is performed. " << endl
         << "\t\t\t"<< "Can be: 'NN_L2, 'RBFN', 'LWR', 'LWPR', 'iRFRLS', 'GPR', 'GMR', 'cocotte', 'cocotte_classifier_Linf' or 'cocotte_classifier_relevant_Linf'." << endl
         << "\t\t\t"<< "Can also be 'all' to activate all methods (default: 'all')" << endl;
}


bool parseArguments(int argc, char *argv[],
                    string& trainingDataFile, string& testDataFile,
                    string& trainingDataStructureFile, string& testDataStructureFile, string& outputFile,
                    bool& defaultBatchSize, unsigned int& batchSize,
                    unsigned int& firstBatchID, bool& defaultLastBatchID, unsigned int& lastBatchID,
                    unordered_map<string, unsigned int> const& methodIDs, vector<bool>& isMethodUsed)
{
    bool showUsage = false, methodOptionUsed = false;
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
            if (trainingDataStructureFile.empty())
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
                showUsage = true;
            }
        }
        else    // option
        {
            if ((current == string("-h")) || (current == string("--help")))
            {
                continue;
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
            else
            {
                cerr << "Unkown option " << current << endl;
            }

            showUsage = true;
        }
    }

    if (trainingDataFile.empty())
    {
        cerr << "Missing training data file name" << endl;
        showUsage = true;
    }

    if (trainingDataStructureFile.empty())
    {
        cerr << "Training data structure file name" << endl;
        showUsage = true;
    }

    if (testDataFile.empty())
    {
        cerr << "Missing test data file name" << endl;
        showUsage = true;
    }

    if (testDataStructureFile.empty())
    {
        testDataStructureFile = trainingDataStructureFile;
    }

    if (!methodOptionUsed)
    {
        for (auto const& item : methodIDs)
        {
            isMethodUsed[item.second] = true;
        }
    }

    return showUsage;
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
        distance = max(distance, temp);
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


void addLine(string outputFileName, unsigned int methodID,
             unsigned int nbTrainingPoints, unsigned int nbTestPoints,
             vector<unsigned int> nbWrongPredictions)
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
        }

        outputFile << endl;
    }

    // Then we write the data
    outputFile << methodID << ", "
               << nbTrainingPoints << ", "
               << nbTestPoints;

    for (auto const& dim : nbWrongPredictions)
    {
        outputFile << ", " << dim;
    }

    outputFile << endl;
}






