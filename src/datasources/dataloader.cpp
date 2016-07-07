
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
#include <cstdlib>
using std::atof;
#include <cctype>
using std::isspace;
#include <vector>
using std::vector;
#include <unordered_map>
using std::unordered_map;
using std::pair;
#include <set>
using std::set;
#include <string>
using std::string;
using std::stod;
#include <sstream>
using std::stringstream;
#include <fstream>
using std::ifstream;
#include <cocotte/datatypes.h>
using Cocotte::Measure;
using Cocotte::DataPoint;
#include <datasources/dataloader.h>



namespace DataSources {



// Constructor
// Default values to load training data,
// add the variable names to load test data
DataLoader::DataLoader(string dataFileName,
                       string structureFileName,
                       std::vector<std::string> const& inputNames,
                       std::vector<std::vector<std::string>> const& outputNames):
    inputVariableNames(inputNames), outputVariableNames(outputNames)
{
    bool const loadingTestData = !outputNames.empty();

    set<string> usedVars;

    unordered_map<string, double> varToFixedPrecision;
    unordered_map<string, string> varToPrecision;
    unordered_map<string, string> precisionToVar;

    unordered_map<string, int> inputIDs;
    unordered_map<string, pair<int,int>> outputIDs;

    ////////////////////////////
    // Loading data structure //
    ////////////////////////////
    {
        ifstream structureSource(structureFileName);
        if (!structureSource.is_open())
        {
            cerr << "Error: Failed to open file " << structureFileName << endl;
            exit(1);
        }

        string line;
        vector<string> cut;

        // We remove the possible empty lines at the top of the file
        while (cut.empty())
        {
            if (!getline(structureSource, line))
            {
                cerr << "Error: No line to read in data structure file " << structureFileName << endl;
                exit(1);
            }

            cut = cutString(line);
        }

        // We read the variables declaration
        while (!cut.empty())
        {
            if (cut.size() > 2)
            {
                cerr << "Error: Wrong syntax for variables declaration in data structure file " << structureFileName << endl;
                exit(1);
            }

            usedVars.insert(cut[0]);

            if (cut.size() == 2)
            {
                if (isDouble(cut[1]))
                {
                    varToFixedPrecision.emplace(cut[0], stod(cut[1]));
                }
                else
                {
                    varToPrecision.emplace(cut[0], cut[1]);
                    precisionToVar.emplace(cut[1], cut[0]);
                }
            }

            if (!getline(structureSource, line))
            {
                cerr << "Error: Missing line(s) after variables declaration in data structure file " << structureFileName << endl;
                exit(1);
            }

            cut = cutString(line);
        }

        // We remove the empty lines between the variables declaration and the outputs declaration
        while (cut.empty())
        {
            if (!getline(structureSource, line))
            {
                cerr << "Error: No declared output in data structure file " << structureFileName << endl;
                exit(1);
            }

            cut = cutString(line);
        }

        // Inputs and outputs
        if (loadingTestData)
        {
            // Input variables order
            for (size_t i = 0; i < inputVariableNames.size(); ++i)
            {
                inputIDs.emplace(inputVariableNames[i],i);
            }

            // Output variables order
            for (size_t i = 0; i < outputVariableNames.size(); ++i)
            {
                auto const& out = outputVariableNames[i];
                for (size_t j = 0; j < out.size(); ++j)
                {
                    outputIDs.emplace(out[j], pair<int,int>(i, j));
                }
            }
        }
        else
        {
            // We read the outputs declaration
            while (!cut.empty())
            {
                int const id = outputVariableNames.size();
                outputVariableNames.push_back(cut);

                for (size_t i = 0; i < cut.size(); ++i)
                {
                    string const name = cut[i];
                    usedVars.insert(name);
                    outputIDs.emplace(name, pair<int,int>(id, i));
                }

                if (!getline(structureSource, line))
                {
                    break;
                }

                cut = cutString(line);
            }

            // We list input variables
            for (auto const& name : usedVars)
            {
                if (outputIDs.count(name) == 0)
                {
                    int const id = inputVariableNames.size();
                    inputIDs.emplace(name, id);
                    inputVariableNames.push_back(name);
                }
            }
        }
    }



    /////////////////////////////////
    // Extracting Some Information //
    /////////////////////////////////

    int const nbInputs = inputVariableNames.size();
    int const nbOutputs = outputVariableNames.size();
    vector<int> outputDimensions(nbOutputs);

    vector<bool> fixedPrecisionInput(nbInputs, false);
    vector<double> inputPrecisions(nbInputs, 0.);
    vector<vector<bool>> fixedPrecisionOutput(nbOutputs);
    vector<vector<double>> outputPrecisions(nbOutputs);

    for (int i = 0; i < nbInputs; ++i)
    {
        string const name = inputVariableNames[i];
        if (varToPrecision.count(name) == 0)
        {
            fixedPrecisionInput[i] = true;

            if (varToFixedPrecision.count(name) == 0)
            {
                // ATTENTION: If input precision is used in the future, this will have to be changed
                fixedPrecisionInput[i] = 0.;
            }
            else
            {
                fixedPrecisionInput[i] = varToFixedPrecision.at(name);
            }
        }
    }

    for (int i = 0; i < nbOutputs; ++i)
    {
        int const nbDims = outputVariableNames[i].size();
        outputDimensions[i] = nbDims;
        vector<bool> fixed(nbDims, false);
        vector<double> prec(nbDims, 0.);

        for (int j = 0; j < nbDims; ++j)
        {
            string const name = outputVariableNames[i][j];
            if (varToPrecision.count(name) == 0)
            {
                fixed[j] = true;

                if (varToFixedPrecision.count(name) == 0)
                {
                    cerr << "Error: No precision for one of the outputs" << endl;
                    exit(1);
                }
                else
                {
                    prec[j] = varToFixedPrecision.at(name);
                }
            }
        }

        fixedPrecisionOutput[i] = fixed;
        outputPrecisions[i] = prec;
    }



    //////////////////
    // Loading data //
    //////////////////

    ifstream dataSource(dataFileName);
    string line, item;
    size_t nbColumns;

    if (!dataSource.is_open())
    {
        cerr << "Error: Failed to open file " << dataFileName << endl;
        exit(1);
    }

    vector<int> inputColumnID(nbInputs);
    vector<int> inputPrecisionColumnID(nbInputs);
    vector<vector<int>> outputColumnID(nbOutputs);
    vector<vector<int>> outputPrecisionColumnID(nbOutputs);

    // Identifying columns
    {
        for (int i = 0; i < nbOutputs; ++i)
        {
            outputColumnID[i].resize(outputDimensions[i]);
            outputPrecisionColumnID[i].resize(outputDimensions[i]);
        }

        if (!getline(dataSource, line))
        {
            cerr << "Error: No line to read in training data file " << dataFileName << endl;
            exit(1);
        }

        stringstream lineStream(line);

        // Listing column names
        vector<string> names;
        while (getline(lineStream, item, ','))
        {
            names.push_back((cutString(item))[0]);
        }

        nbColumns = names.size();
        for (size_t i = 0; i < nbColumns; ++i)
        {
            string name = names[i];
            bool isPrec = false;

            if (precisionToVar.count(name) > 0)
            {
                isPrec = true;
                name = precisionToVar.at(name);
            }

            if (usedVars.count(name) == 0)
            {
                    continue;
            }

            if (inputIDs.count(name) > 0)
            {
                if (isPrec)
                {
                    inputPrecisionColumnID[inputIDs.at(name)] = i;
                }
                else
                {
                    inputColumnID[inputIDs.at(name)] = i;
                }
            }
            else
            {
                pair<int,int> const IDs = outputIDs.at(name);
                if (isPrec)
                {
                    outputPrecisionColumnID[IDs.first][IDs.second] = i;
                }
                else
                {
                    outputColumnID[IDs.first][IDs.second] = i;
                }
            }
        }
    }

    // Reading data
    while (getline(dataSource, line))
    {
        stringstream lineStream(line);
        vector<double> vals;
        vals.reserve(nbColumns);

        while (getline(lineStream, item, ','))
        {
            vals.push_back(stod(item));
        }

        if (vals.size() != nbColumns)
        {
            cerr << "Error: inconsistent number of columns in training data file " << dataFileName << endl;
            exit(1);
        }

        DataPoint point;
        point.x.reserve(nbInputs);

        for (int i = 0; i < nbInputs; ++i)
        {
            if (fixedPrecisionInput[i])
            {
                point.x.push_back(Measure{vals[inputColumnID[i]],
                                          inputPrecisions[i]});
            }
            else
            {
                point.x.push_back(Measure{vals[inputColumnID[i]],
                                          vals[inputPrecisionColumnID[i]]});
            }
        }

        point.t.reserve(nbOutputs);
        for (int i = 0; i < nbOutputs; ++i)
        {
            vector<Measure> out;

            int const outputSize = outputDimensions[i];
            out.reserve(outputSize);

            for (int j = 0; j < outputSize; ++j)
            {
                if (fixedPrecisionOutput[i][j])
                {
                    out.push_back(Measure{vals[outputColumnID[i][j]],
                                              outputPrecisions[i][j]});
                }
                else
                {
                    out.push_back(Measure{vals[outputColumnID[i][j]],
                                              vals[outputPrecisionColumnID[i][j]]});
                }
            }

            point.t.push_back(out);
        }

        loadedData.push_back(point);
    }

    // Checking that there was data to read
    if (loadedData.size() < 1)
    {
        cerr << "Error: No data to read in training data file " << dataFileName << endl;
        exit(1);
    }

    if (loadingTestData)
    {
        {
            size_t const nbPoints = loadedData.size();
            testDataInput.reserve(nbPoints);
            testDataOutput.reserve(nbPoints);
            testDataOutputPrecisions.reserve(nbPoints);
        }

        for (auto const& point : loadedData)
        {
            // Input
            {
                vector<double> x;
                x.reserve(point.x.size());
                for (auto const& dim : point.x)
                {
                    x.push_back(dim.value);
                }
                testDataInput.push_back(x);
            }

            // Outputs
            {
                vector<vector<double>> t;
                t.reserve(point.t.size());
                vector<vector<double>> tPrec;
                tPrec.reserve(point.t.size());

                for (auto const& output : point.t)
                {
                    vector<double> out;
                    out.reserve(output.size());
                    vector<double> outPrec;
                    outPrec.reserve(output.size());

                    for (auto const& dim : output)
                    {
                        out.push_back(dim.value);
                        outPrec.push_back(dim.precision);
                    }
                    t.push_back(out);
                    tPrec.push_back(outPrec);
                }
                testDataOutput.push_back(t);
                testDataOutputPrecisions.push_back(tPrec);
            }
        }

        loadedData.resize(0);
    }
}



// Main methods
DataPoint DataLoader::getTrainingDataPoint()
{
    if (loadedData.empty())
    {
        cerr << "Error: Not enough training data points have been loaded" << endl;
        exit(1);
    }

    DataPoint const result = loadedData.back();
    loadedData.pop_back();

    return result;
}


vector<DataPoint> DataLoader::getTrainingDataPoints(int nbDataPoints)
{
    int const before = loadedData.size();

    if (before < nbDataPoints)
    {
        cerr << "Error: Not enough training data points have been loaded" << endl;
        exit(1);
    }

    int const after = before - nbDataPoints;
    vector<DataPoint> const result(loadedData.begin() + after, loadedData.end());
    loadedData.resize(after);

    return result;
}


pair<vector<vector<double>>, pair<vector<vector<vector<double>>>, vector<vector<vector<double>>>>> DataLoader::getTestDataPoints(int nbDataPoints)
{
    int const before = testDataInput.size();

    if (before < nbDataPoints)
    {
        cerr << "Error: Not enough test data points have been loaded" << endl;
        exit(1);
    }

    int const after = before - nbDataPoints;
    vector<vector<double>> const inputs(testDataInput.begin() + after, testDataInput.end());
    testDataInput.resize(after);
    vector<vector<vector<double>>> const outputs(testDataOutput.begin() + after, testDataOutput.end());
    testDataOutput.resize(after);
    vector<vector<vector<double>>> const outputPrecs(testDataOutputPrecisions.begin() + after, testDataOutputPrecisions.end());
    testDataOutputPrecisions.resize(after);

    return {inputs, {outputs, outputPrecs}};
}


vector<string> DataLoader::getInputVariableNames()
{
    return inputVariableNames;
}


vector<vector<string>> DataLoader::getOutputVariableNames()
{
    return outputVariableNames;
}



// Helper methods
vector<string> DataLoader::cutString(string line)
{
    vector<string> result;
    auto it = line.begin(), end = line.end();

    while (true)
    {
        while ((it != end) && isspace(*it))
        {
            ++it;
        }

        auto const first = it;

        while ((it != end) && !isspace(*it))
        {
            ++it;
        }

        if (first != it)
        {
            result.push_back(string(first, it));
        }

        if (it == end)
        {
            break;
        }
    }

    return result;
}


bool DataLoader::isDouble(string item)
{
    stringstream itemStream(item);
    double d; char c;

    return ((itemStream >> d) && !(itemStream >> c));
}

}

