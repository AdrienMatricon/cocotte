
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
#include <list>
using std::list;
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
    list<string> usedVarsInOrder;

    unordered_map<string, double> varNameToFixedPrecision;
    unordered_map<string, string> varNameToPrecisionName;
    unordered_map<string, string> precisionNameToVarName;

    unordered_map<string, unsigned int> inputIDs;
    unordered_map<string, pair<unsigned int, unsigned int>> outputIDs;

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
            usedVarsInOrder.push_back(cut[0]);

            if (cut.size() == 2)
            {
                if (isDouble(cut[1]))
                {
                    varNameToFixedPrecision.emplace(cut[0], stod(cut[1]));
                }
                else
                {
                    varNameToPrecisionName.emplace(cut[0], cut[1]);
                    precisionNameToVarName.emplace(cut[1], cut[0]);
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
            for (unsigned int i = 0; i < inputVariableNames.size(); ++i)
            {
                inputIDs.emplace(inputVariableNames[i],i);
            }

            // Output variables order
            for (unsigned int i = 0; i < outputVariableNames.size(); ++i)
            {
                auto const& out = outputVariableNames[i];
                for (unsigned int j = 0; j < out.size(); ++j)
                {
                    outputIDs.emplace(out[j], pair<unsigned int, unsigned int>(i, j));
                }
            }
        }
        else
        {
            // We read the outputs declaration
            while (!cut.empty())
            {
                unsigned int const id = outputVariableNames.size();
                outputVariableNames.push_back(cut);

                for (unsigned int i = 0; i < cut.size(); ++i)
                {
                    string const name = cut[i];
                    usedVars.insert(name);
                    usedVarsInOrder.push_back(name);
                    outputIDs.emplace(name, pair<unsigned int, unsigned int>(id, i));
                }

                if (!getline(structureSource, line))
                {
                    break;
                }

                cut = cutString(line);
            }

            // We list input variables
            for (auto const& name : usedVarsInOrder)
            {
                if (outputIDs.count(name) == 0)
                {
                    unsigned int const id = inputVariableNames.size();
                    inputIDs.emplace(name, id);
                    inputVariableNames.push_back(name);
                }
            }
        }
    }



    /////////////////////////////////
    // Extracting Some Information //
    /////////////////////////////////

    unsigned int const nbInputs = inputVariableNames.size();
    unsigned int const nbOutputs = outputVariableNames.size();
    vector<unsigned int> outputDimensions(nbOutputs);

    vector<bool> isFixedPrecisionInput(nbInputs, false);
    vector<double> inputIDToFixedPrecision(nbInputs, 0.);
    vector<vector<bool>> isFixedPrecisionOutput(nbOutputs);
    vector<vector<double>> outputIDToFixedPrecision(nbOutputs);

    for (unsigned int i = 0; i < nbInputs; ++i)
    {
        string const name = inputVariableNames[i];
        if (varNameToPrecisionName.count(name) == 0)
        {
            // If there is no precision variable, then the input precision is fixed
            isFixedPrecisionInput[i] = true;

            if (varNameToFixedPrecision.count(name) != 0)
            {
                // If a fixed precision is specified, we use it
                // Otherwise we leave it at 0 (infinite precision)
                inputIDToFixedPrecision[i] = varNameToFixedPrecision.at(name);
            }
        }
    }

    for (unsigned int i = 0; i < nbOutputs; ++i)
    {
        unsigned int const nbDims = outputVariableNames[i].size();
        outputDimensions[i] = nbDims;
        vector<bool> fixed(nbDims, false);
        vector<double> prec(nbDims, 0.);

        for (unsigned int j = 0; j < nbDims; ++j)
        {
            string const name = outputVariableNames[i][j];
            if (varNameToPrecisionName.count(name) == 0)
            {
                fixed[j] = true;

                if (varNameToFixedPrecision.count(name) == 0)
                {
                    cerr << "Error: No precision for one of the outputs" << endl;
                    exit(1);
                }
                else
                {
                    prec[j] = varNameToFixedPrecision.at(name);
                }
            }
        }

        isFixedPrecisionOutput[i] = fixed;
        outputIDToFixedPrecision[i] = prec;
    }



    //////////////////
    // Loading data //
    //////////////////

    ifstream dataSource(dataFileName);
    string line, item;
    unsigned int nbColumns;

    if (!dataSource.is_open())
    {
        cerr << "Error: Failed to open file " << dataFileName << endl;
        exit(1);
    }

    vector<unsigned int> inputColumnID(nbInputs);
    vector<unsigned int> inputPrecisionColumnID(nbInputs);
    vector<vector<unsigned int>> outputColumnID(nbOutputs);
    vector<vector<unsigned int>> outputPrecisionColumnID(nbOutputs);

    // Identifying columns
    {
        for (unsigned int i = 0; i < nbOutputs; ++i)
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
        for (unsigned int i = 0; i < nbColumns; ++i)
        {
            string name = names[i];
            bool isPrec = false;

            if (precisionNameToVarName.count(name) > 0)
            {
                isPrec = true;
                name = precisionNameToVarName.at(name);
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
                pair<unsigned int, unsigned int> const IDs = outputIDs.at(name);
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

        for (unsigned int i = 0; i < nbInputs; ++i)
        {
            if (isFixedPrecisionInput[i])
            {
                point.x.push_back(Measure{vals[inputColumnID[i]],
                                          inputIDToFixedPrecision[i]});
            }
            else
            {
                point.x.push_back(Measure{vals[inputColumnID[i]],
                                          vals[inputPrecisionColumnID[i]]});
            }
        }

        point.t.reserve(nbOutputs);
        for (unsigned int i = 0; i < nbOutputs; ++i)
        {
            vector<Measure> out;

            unsigned int const outputSize = outputDimensions[i];
            out.reserve(outputSize);

            for (unsigned int j = 0; j < outputSize; ++j)
            {
                if (isFixedPrecisionOutput[i][j])
                {
                    out.push_back(Measure{vals[outputColumnID[i][j]],
                                              outputIDToFixedPrecision[i][j]});
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
            unsigned int const nbPoints = loadedData.size();
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


vector<DataPoint> DataLoader::getTrainingDataPoints(unsigned int nbDataPoints)
{
    unsigned int const before = loadedData.size();

    if (before < nbDataPoints)
    {
        cerr << "Error: Not enough training data points have been loaded" << endl;
        exit(1);
    }

    unsigned int const after = before - nbDataPoints;
    vector<DataPoint> const result(loadedData.begin() + after, loadedData.end());
    loadedData.resize(after);

    return result;
}


TestData DataLoader::getTestDataPoints(unsigned int nbDataPoints)
{
    unsigned int const before = testDataInput.size();

    if (before < nbDataPoints)
    {
        cerr << "Error: Not enough test data points have been loaded" << endl;
        exit(1);
    }

    unsigned int const after = before - nbDataPoints;
    vector<vector<double>> const inputs(testDataInput.begin() + after, testDataInput.end());
    testDataInput.resize(after);
    vector<vector<vector<double>>> const outputs(testDataOutput.begin() + after, testDataOutput.end());
    testDataOutput.resize(after);
    vector<vector<vector<double>>> const outputPrecs(testDataOutputPrecisions.begin() + after, testDataOutputPrecisions.end());
    testDataOutputPrecisions.resize(after);

    return {inputs, outputs, outputPrecs};
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


unsigned int DataLoader::getNbDataPoints()
{
    if (loadedData.empty())
    {
        // Test datapoints
        return testDataOutput.size();
    }
    else
    {
        // Training datapoints
        return loadedData.size();
    }
}

}

