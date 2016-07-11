
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
#include <cocotte/datatypes.h>
using Cocotte::DataPoint;
#include <datasources/dataloader.h>
using DataSources::DataSource;
using DataSources::DataLoader;
#include <cocotte/approximators/polynomial.h>
using Cocotte::Approximators::Polynomial;
#include <cocotte/learner.h>
using Cocotte::Learner;

#include <soplex/src/soplex.h>


vector<vector<size_t>> computeErrors(vector<vector<vector<double>>> const& estimations,
                                           vector<vector<vector<double>>> const& targets,
                                           vector<vector<vector<double>>> const& targetsPrec)
{
    vector<vector<size_t>> results;
    size_t const nbPoints = estimations.size();
    size_t const nbOutputs = estimations.front().size();
    results.reserve(nbOutputs);
    for (size_t i = 0; i < nbOutputs; ++i)
    {
        results.push_back(vector<size_t>(estimations[0][i].size(), 0.));
    }

    for (size_t k = 0; k < nbPoints; ++k)
        for (size_t i = 0; i < nbOutputs; ++i)
        {
            size_t const nbDims = results[i].size();
            for (size_t j = 0; j < nbDims; ++j)
            {
                if (abs((estimations[k][i][j] - targets[k][i][j]) / targetsPrec[k][i][j]) > 1.)
                {
                    results[i][j] += 1;
                }
            }
        }

    return results;
}


int main(int argc, char *argv[])
{
    if (argc > 1)
    {
        string action = argv[1];

        if (action == "learn")
        {
            if (argc >= 6)
            {

                string trainingData = argv[2];
                string dataStructure = argv[3];
                int const nbPoints = std::atoi(argv[4]);

                int const nbBatches = argc - 5;
                vector<string> outputFiles;
                outputFiles.reserve(nbBatches);
                for (int i = 0; i < nbBatches; ++i)
                {
                    outputFiles.push_back(argv[i+5]);
                }

                int const nbPointsPerBatch = nbPoints/nbBatches;
                int const nbPointsLastBatch = nbPoints - (nbPointsPerBatch * (nbBatches-1));

                DataSource* source = new DataLoader(trainingData, dataStructure);
                cout << "Data loaded." << endl;
                Learner<Polynomial> learner(source->getInputVariableNames(), source->getOutputVariableNames());

                int const lastBatch = nbBatches - 1;
                for (int i = 0; i < lastBatch; ++i)
                {
                    cout << "Adding batch " << i + 1 << "/" << nbBatches << ". Computing..."; cout.flush();
                    learner.addDataPointsIncremental(source->getTrainingDataPoints(nbPointsPerBatch));
                    cout << "done." << endl << "Dumping data... "; cout.flush();
                    learner.dumpModels(outputFiles[i]);
                    cout << "done." << endl;
                }

                cout << "Adding batch " << nbBatches << "/" << nbBatches << ". Computing..."; cout.flush();
                learner.addDataPointsIncremental(source->getTrainingDataPoints(nbPointsLastBatch));
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
            if ((argc >= 6) && (argc <= 7))
            {
                int nbPoints = 10000;

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
                DataSource* source = new DataLoader(inputFile, dataStructure, learner.getInputNames(), learner.getOutputNames());
                cout << "done." << endl;

                cout << "Computing..." << endl;

                auto points = source->getTestDataPoints(nbPoints);
                vector<vector<int>> modelIDs;
                auto estimates = learner.predict(points.first, &modelIDs);
                cout << "Outputs predicted. Dumping data... "; cout.flush();
                learner.dumpEstimates(outputFile, points.first, estimates, points.second.first, modelIDs);
                cout << "done." << endl;
                return 0;
            }
        }
        else if (action == "evaluate")
        {
            if (argc >= 7)
            {
                string inputFile = argv[2];
                string dataStructure = argv[3];
                int const nbPoints = std::atoi(argv[4]);
                string outputFile = argv[5];

                int const nbModelsFiles = argc - 6;
                vector<string> modelsFiles;
                modelsFiles.reserve(nbModelsFiles);

                for (int i = 0; i < nbModelsFiles; ++i)
                {
                    modelsFiles.push_back(argv[i+6]);
                }

                // First models file
                cout << "Loading models (1/" << nbModelsFiles << ")... "; cout.flush();
                Learner<Polynomial> learner(modelsFiles[0]);
                cout << "done." << endl;

                // First line that will be dumped
                auto const outputNames = learner.getOutputNames();
                size_t const nbOutputs = outputNames.size();
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
                DataSource* source = new DataLoader(inputFile, dataStructure, learner.getInputNames(), outputNames);
                auto points = source->getTestDataPoints(nbPoints);
                vector<vector<double>> const& inputs = points.first;
                vector<vector<vector<double>>> const& targets = points.second.first;
                vector<vector<vector<double>>> const& targetsPrec = points.second.second;
                cout << "done." << endl;

                cout << "Processing... "; cout.flush();
                auto errors = computeErrors(learner.predict(inputs), targets, targetsPrec);
                cout << "done." << endl;

                // New line to be dumped
                result << learner.getNbPoints() << ", " << nbPoints;
                for (size_t i = 0; i < nbOutputs; ++i)
                {
                    size_t const nbDims = errors[i].size();
                    for (size_t j = 0; j < nbDims; ++j)
                    {
                        result << ", " << learner.getComplexity(i,j);
                        result << ", " << errors[i][j];
                    }
                }
                result << endl;

                for (int k = 1; k < nbModelsFiles; ++k)
                {
                    cout << "Repeating (" << k+1 << "/" << nbModelsFiles << ")... "; cout.flush();
                    learner = Learner<Polynomial>(modelsFiles[k]);
                    errors = computeErrors(learner.predict(inputs), targets, targetsPrec);

                    // New line to be dumped
                    result << learner.getNbPoints() << ", " << nbPoints;
                    for (size_t i = 0; i < nbOutputs; ++i)
                    {
                        size_t const nbDims = errors[i].size();
                        for (size_t j = 0; j < nbDims; ++j)
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

    cerr << "Usage: cocotte learn <training data file> <data structure file> <nb points for training> <output file for models> [other output files for models]" << endl;
    cerr << "Usage: cocotte test <models data file> <test data file> <data structure file> <output file> [nb points to predict=10000]" << endl;
    cerr << "Usage: cocotte evaluate <input data file> <data structure file> <nb points for testing> <outputFile> <models data file> [other models data file]" << endl;
    return 1;
}
