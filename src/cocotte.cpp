
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
#include <cstdlib>
using std::atoi;
#include <cmath>
using std::abs;
#include <string>
using std::string;
using std::to_string;
#include <sstream>
using std::stringstream;
#include <fstream>
using std::ofstream;
#include <vector>
using std::vector;
#include <memory>
using std::unique_ptr;
#include <cocotte/datatypes.h>
using Cocotte::DataPoint;
#include <datasources/dataloader.h>
using DataSources::DataSource;
using DataSources::DataLoader;
#include <cocotte/approximators/polynomial.h>
using Cocotte::Approximators::Polynomial;
#include <cocotte/learner.h>
using Cocotte::Learner;


vector<vector<unsigned int>> computeErrors(vector<vector<vector<double>>> const& estimations,
                                           vector<vector<vector<double>>> const& targets,
                                           vector<vector<vector<double>>> const& targetsPrec)
{
    vector<vector<unsigned int>> results;
    unsigned int const nbPoints = estimations.size();
    unsigned int const nbOutputs = estimations.front().size();
    results.reserve(nbOutputs);
    for (unsigned int i = 0; i < nbOutputs; ++i)
    {
        results.push_back(vector<unsigned int>(estimations[0][i].size(), 0.));
    }

    for (unsigned int k = 0; k < nbPoints; ++k)
        for (unsigned int i = 0; i < nbOutputs; ++i)
        {
            unsigned int const nbDims = results[i].size();
            for (unsigned int j = 0; j < nbDims; ++j)
            {
                if (abs((estimations[k][i][j] - targets[k][i][j]) / targetsPrec[k][i][j]) > 1.)
                {
                    results[i][j] += 1;
                }
            }
        }

    return results;
}


void dumpEstimates(string fileName,
                   vector<string> inputNames,
                   vector<vector<string>> outputNames,
                   vector<vector<double>> const& x,
                   vector<vector<vector<double>>> const& estimates,
                   vector<vector<vector<double>>> const& actual = vector<vector<vector<double>>>(0),
                   vector<vector<unsigned int>> const& modelIDs = vector<vector<unsigned int>>(0))
{
    ofstream outputFile(fileName);
    if (!outputFile.is_open())
    {
        cerr << "Failed to open file" << endl;
        exit(1);
    }


    unsigned int const nbInputs = inputNames.size();
    unsigned int const nbOutputs = outputNames.size();
    unsigned int const nbDataPoints = x.size();
    bool const knownActual = (actual.size() > 0);
    bool const dumpModelIDs = (modelIDs.size() > 0);

    outputFile << inputNames[0];
    for (unsigned int i = 1; i < nbInputs; ++i)
    {
        outputFile << ", " << inputNames[i];
    }

    for (auto& out : outputNames)
    {
        for (auto& name : out)
        {
            outputFile << ", " << name;
        }
    }

    if (knownActual)
    {
        for (auto& out : outputNames)
        {
            for (auto& name : out)
            {
                outputFile << ", " << name << "_actual";
            }
        }
    }

    if (dumpModelIDs)
    {
        for (unsigned int i = 0; i < nbOutputs; ++i)
        {
            outputFile << ", output" << i << "_mID";
        }
    }

    outputFile << endl;

    for (unsigned int i = 0; i < nbDataPoints; ++i)
    {
        outputFile << x[i][0];
        for (unsigned int j = 1; j < nbInputs; ++j)
        {
            outputFile << ", " << x[i][j];
        }

        for (auto& out : estimates[i])
        {
            for (auto& est : out)
            {
                outputFile << ", " << est;
            }
        }

        if (knownActual)
        {
            for (auto& out : actual[i])
            {
                for (auto& act : out)
                {
                    outputFile << ", " << act;
                }
            }
        }

        if (dumpModelIDs)
        {
            for (auto& mID : modelIDs)
            {
                outputFile << ", " << mID[i];
            }
        }

        outputFile << endl;
    }
}


int main(int argc, char *argv[])
{
    bool showAll = true;
    string action;

    if (argc > 1)
    {
        action = argv[1];

        if (action == "learn")
        {
            showAll = false;
            if (argc >= 6)
            {
                string mode = "normal";

                string trainingData = argv[2];
                string dataStructure = argv[3];
                unsigned int const nbPoints = std::atoi(argv[4]);

                unsigned int const nbBatches = argc - 5;
                vector<string> outputFiles;
                outputFiles.reserve(nbBatches);
                for (unsigned int i = 0; i < nbBatches; ++i)
                {
                    outputFiles.push_back(argv[i+5]);
                }

                unsigned int const nbPointsPerBatch = nbPoints/nbBatches;
                unsigned int const nbPointsLastBatch = nbPoints - (nbPointsPerBatch * (nbBatches-1));

                unique_ptr<DataSource> source(new DataLoader(trainingData, dataStructure));
                cout << "Data loaded." << endl;
                Learner<Polynomial> learner(source->getInputVariableNames(), source->getOutputVariableNames());

                unsigned int const lastBatch = nbBatches - 1;
                for (unsigned int i = 0; i < lastBatch; ++i)
                {
                    cout << "Adding batch " << i + 1 << "/" << nbBatches << ". Computing..."; cout.flush();

                    if (mode == "incremental")
                    {
                        learner.addDataPointsIncremental(source->getTrainingDataPoints(nbPointsPerBatch));
                    }
                    else if (mode == "no rollback")
                    {
                        learner.addDataPointsNoRollback(source->getTrainingDataPoints(nbPointsPerBatch));
                    }
                    else
                    {
                        learner.addDataPoints(source->getTrainingDataPoints(nbPointsPerBatch));
                    }

                    cout << "done." << endl << "Dumping data... "; cout.flush();
                    learner.dumpModels(outputFiles[i]);
                    cout << "done." << endl;
                }

                cout << "Adding batch " << nbBatches << "/" << nbBatches << ". Computing..."; cout.flush();

                if (mode == "incremental")
                {
                    learner.addDataPointsIncremental(source->getTrainingDataPoints(nbPointsLastBatch));
                }
                else if (mode == "no rollback")
                {
                    learner.addDataPointsNoRollback(source->getTrainingDataPoints(nbPointsLastBatch));
                    cout << "done." << endl;
                    cout << "Restructuring..."; cout.flush();
                    learner.restructureModels();
                    cout << "done." << endl;
                    cout << "Removing artifacts..."; cout.flush();
                    learner.removeArtifacts();
                }
                else
                {
                    learner.addDataPoints(source->getTrainingDataPoints(nbPointsLastBatch));
                    cout << "done." << endl;
                    cout << "Removing artifacts..."; cout.flush();
                    learner.removeArtifacts();
                }

                cout << "done." << endl << endl;
                cout << learner << endl << endl;
                cout << "Dumping data... "; cout.flush();
                learner.dumpModels(outputFiles[lastBatch]);
                cout << "done." << endl;
                return 0;
            }
        }
        else if (action == "test")
        {
            showAll = false;
            if ((argc >= 6) && (argc <= 7))
            {
                unsigned int nbPoints = 10000;

                if (argc == 7)
                {
                    nbPoints = std::atoi(argv[6]);
                }

                string modelsFile = argv[2];
                string inputFile = argv[3];
                string dataStructure = argv[4];
                string outputFile = argv[5];

                cout << "Loading models... "; cout.flush();
                Learner<Polynomial> learner(modelsFile);
                cout << "done." << endl << endl;
                cout << learner << endl << endl;

                cout << "Loading input data... "; cout.flush();
                unique_ptr<DataSource> source(new DataLoader(inputFile, dataStructure, learner.getInputNames(), learner.getOutputNames()));
                cout << "done." << endl;

                cout << "Computing..." << endl;

                auto points = source->getTestDataPoints(nbPoints);
                vector<vector<unsigned int>> modelIDs;
                auto estimates = learner.predict(points.xValues, &modelIDs);
                cout << "Outputs predicted. Dumping data... "; cout.flush();
                dumpEstimates(outputFile, learner.getInputNames(), learner.getOutputNames(), points.xValues, estimates, points.tValues, modelIDs);
                cout << "done." << endl;
                return 0;
            }
        }
        else if (action == "evaluate")
        {
            showAll = false;
            if (argc >= 7)
            {
                string inputFile = argv[2];
                string dataStructure = argv[3];
                unsigned int const nbPoints = std::atoi(argv[4]);
                string outputFile = argv[5];

                unsigned int const nbModelsFiles = argc - 6;
                vector<string> modelsFiles;
                modelsFiles.reserve(nbModelsFiles);

                for (unsigned int i = 0; i < nbModelsFiles; ++i)
                {
                    modelsFiles.push_back(argv[i+6]);
                }

                // First models file
                cout << "Loading models (1/" << nbModelsFiles << ")... "; cout.flush();
                Learner<Polynomial> learner(modelsFiles[0]);
                cout << "done." << endl;

                // First line that will be dumped
                auto const outputNames = learner.getOutputNames();
                unsigned int const nbOutputs = outputNames.size();
                stringstream result;
                result << "nbTrainingPoints, nbTestPoints";
                for (auto const& output : outputNames)
                {
                    for (auto const& dim : output)
                    {
                        result << ", " << dim << "_comp";
                        result << ", " << dim << "_err";
                    }
                }
                result << endl;

                cout << "Loading input data... "; cout.flush();
                unique_ptr<DataSource> source(new DataLoader(inputFile, dataStructure, learner.getInputNames(), outputNames));
                auto points = source->getTestDataPoints(nbPoints);
                vector<vector<double>> const& inputs = points.xValues;
                vector<vector<vector<double>>> const& targets = points.tValues;
                vector<vector<vector<double>>> const& targetsPrec = points.tPrecisions;
                cout << "done." << endl;

                cout << "Processing... "; cout.flush();
                auto errors = computeErrors(learner.predict(inputs), targets, targetsPrec);
                cout << "done." << endl;

                // New line to be dumped
                result << learner.getNbPoints() << ", " << nbPoints;
                for (unsigned int i = 0; i < nbOutputs; ++i)
                {
                    unsigned int const nbDims = errors[i].size();
                    for (unsigned int j = 0; j < nbDims; ++j)
                    {
                        result << ", " << learner.getComplexity(i,j);
                        result << ", " << errors[i][j];
                    }
                }
                result << endl;

                for (unsigned int k = 1; k < nbModelsFiles; ++k)
                {
                    cout << "Repeating (" << k+1 << "/" << nbModelsFiles << ")... "; cout.flush();
                    learner = Learner<Polynomial>(modelsFiles[k]);
                    errors = computeErrors(learner.predict(inputs), targets, targetsPrec);

                    // New line to be dumped
                    result << learner.getNbPoints() << ", " << nbPoints;
                    for (unsigned int i = 0; i < nbOutputs; ++i)
                    {
                        unsigned int const nbDims = errors[i].size();
                        for (unsigned int j = 0; j < nbDims; ++j)
                        {
                            result << ", " << learner.getComplexity(i,j);
                            result << ", " << errors[i][j];
                        }
                    }
                    result << endl;

                    cout << "done." << endl;
                }

                cout << "Dumping results... "; cout.flush();

                ofstream outputFileStream(outputFile);
                if (!outputFileStream.is_open())
                {
                    cerr << "Failed to open file" << endl;
                    exit(1);
                }

                outputFileStream << result.str();

                cout << "done." << endl;

                return 0;
            }
        }
    }

    if (showAll || (action == "learn"))
    {
        cerr << "Usage: cocotte learn <training data file> <data structure file> <nb points for training> <output file for models> [other output files for models]" << endl;
    }

    if (showAll || (action == "test"))
    {
        cerr << "Usage: cocotte test <models data file> <test data file> <data structure file> <output file> [nb points to predict=10000]" << endl;
    }

    if (showAll || (action == "evaluate"))
    {
        cerr << "Usage: cocotte evaluate <test data file> <data structure file> <nb points for testing> <outputFile> <models data file> [other models data file]" << endl;
    }

    return 1;
}
