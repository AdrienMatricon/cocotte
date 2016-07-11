
#include <vector>
using std::vector;
#include <string>
using std::string;
#include <sstream>
using std::stringstream;
#include <iostream>
using std::cerr;
using std::endl;
#include <fstream>
using std::ifstream;
using std::ofstream;
#include <boost/shared_ptr.hpp>
using boost::shared_ptr;
#include<boost/archive/text_iarchive.hpp>
using boost::archive::text_iarchive;
#include<boost/archive/text_oarchive.hpp>
using boost::archive::text_oarchive;
#include <cocotte/datatypes.h>
using Cocotte::DataPoint;
#include <cocotte/models/models.hh>
#include <cocotte/modellist.h>
#include <cocotte/learner.h>

namespace Cocotte {



// Constructor
template <typename ApproximatorType>
Learner<ApproximatorType>::Learner(vector<string> const& iNames, vector<vector<string>> const& oNames):
    inputNames(iNames), outputNames(oNames), nbOutputs(oNames.size())
{
    int const nbInputDims = iNames.size();
    modelLists.reserve(oNames.size());
    for (int i = 0; i < nbOutputs; ++i)
    {
        modelLists.push_back(ModelList<ApproximatorType>(i, nbInputDims, oNames[i].size()));
    }
}



// Accessors
template <typename ApproximatorType>
vector<string> Learner<ApproximatorType>::getInputNames() const
{
    return inputNames;
}


template <typename ApproximatorType>
vector<vector<string>> Learner<ApproximatorType>::getOutputNames() const
{
    return outputNames;
}


template <typename ApproximatorType>
size_t Learner<ApproximatorType>::getNbPoints() const
{
    return data.size();
}


template <typename ApproximatorType>
size_t Learner<ApproximatorType>::getComplexity(size_t i, size_t j) const
{
    return modelLists[i].getComplexity(j);
}



// Serialization
template <typename ApproximatorType>
Learner<ApproximatorType>::Learner(string fileName)
{
    ifstream inputFile(fileName);
    if (!inputFile.is_open())
    {
        cerr << "Failed to open file" << endl;
        exit(1);
    }

    text_iarchive inputArchive(inputFile);
    inputArchive >> *this;
}


template <typename ApproximatorType>
void Learner<ApproximatorType>::dumpModels(string fileName)
{
    ofstream outputFile(fileName);
    if (!outputFile.is_open())
    {
        cerr << "Failed to open file" << endl;
        exit(1);
    }

    text_oarchive outputArchive(outputFile);
    outputArchive << *this;
}



// Main methods

// Points can be added in any order
template <typename ApproximatorType>
void Learner<ApproximatorType>::addDataPoint(DataPoint const& point)
{
    data.push_back(boost::shared_ptr<DataPoint const>(new DataPoint(point)));

    for (auto& mList : modelLists)
    {
        mList.addPoint(data.back());
    }
}


template <typename ApproximatorType>
void Learner<ApproximatorType>::addDataPoints(vector<DataPoint> const& points)
{
    data.reserve(data.size() + points.size());
    vector<boost::shared_ptr<DataPoint const>> pointers;

    for (auto const& point : points)
    {
        data.push_back(boost::shared_ptr<DataPoint const>(new DataPoint(point)));
        pointers.push_back(data.back());
    }

    for (auto& mList : modelLists)
    {
        mList.addPoints(pointers);
    }
}


// Should be called after adding all points
template <typename ApproximatorType>
void Learner<ApproximatorType>::removeArtifacts()
{
    for (auto& mList: modelLists)
    {
        // For each old model, we check if all its points could all fit in other models
        int const nbModels = mList.getNbModels();

        if (nbModels < 2)
        {
            continue;
        }

        for (int j = 0; j < nbModels; ++j)
        {
            // We remove the first model
            auto firstModel = mList.firstModel();
            mList.removeFirstModel();

            // We save the information we need to roll back what we will do
            auto const oldList = mList;

            // We try to add all points from the current model
            bool success = true;

            for (auto mIt = Models::pointsBegin(firstModel), mEnd = Models::pointsEnd(firstModel);
                 mIt != mEnd; ++mIt)
            {
                if (!mList.tryAddingPointToExistingModels(mIt.getSharedPointer()))
                {
                    success = false;
                    break;
                }
            }

            // If the points were successfully distributed, the model was artifact
            // Otherwise, we roll back the changes, and the first model becomes the last model
            if (!success)
            {
                mList = std::move(oldList);
                mList.addModel(firstModel);
            }
        }
    }
}


// Faster, greedy, the order in which the points are added matters
template <typename ApproximatorType>
void Learner<ApproximatorType>::addDataPointToExistingModels(DataPoint const& point)
{
    data.push_back(boost::shared_ptr<DataPoint const>(new DataPoint(point)));

    for (auto& mList : modelLists)
    {
        mList.tryAddingPointToExistingModels(data.back());
    }
}


template <typename ApproximatorType>
void Learner<ApproximatorType>::addDataPointsToExistingModels(vector<DataPoint> const& points)
{
    for (auto const& point : points)
    {
        addDataPointToExistingModels(point);
    }
}


// Makes it as if all points were added with addDataPoints(), then runs removeArtifacts()
template <typename ApproximatorType>
void Learner<ApproximatorType>::restructureModels()
{
    for (auto& mList : modelLists)
    {
        mList.restructureModels();
    }

    removeArtifacts();
}


// For fully incremental learning
// Rolls back removeArtifacts(), adds the points,
// Then calls removeArtifacts()
template <typename ApproximatorType>
void Learner<ApproximatorType>::addDataPointIncremental(DataPoint const& point)
{
    data.push_back(boost::shared_ptr<DataPoint const>(new DataPoint(point)));

    for (auto& mList : modelLists)
    {
        mList.restructureModels();
        mList.addPoint(data.back());
    }

    removeArtifacts();
}


template <typename ApproximatorType>
void Learner<ApproximatorType>::addDataPointsIncremental(vector<DataPoint> const& points)
{
    data.reserve(data.size() + points.size());
    vector<boost::shared_ptr<DataPoint const>> pointers;

    for (auto const& point : points)
    {
        data.push_back(boost::shared_ptr<DataPoint const>(new DataPoint(point)));
        pointers.push_back(data.back());
    }

    for (auto& mList : modelLists)
    {
        mList.restructureModels();
        mList.addPoints(pointers);
    }

    removeArtifacts();
}


// Predicts outputs for new points
template <typename ApproximatorType>
vector<vector<vector<double>>> Learner<ApproximatorType>::predict(vector<vector<double>> const& x, bool shouldObtainModelIDs, vector<vector<int>> *modelIDs)
{
    int const nbPoints = x.size();
    vector<vector<vector<double>>> rawPredictions;
    rawPredictions.reserve(nbOutputs);

    if (shouldObtainModelIDs)
    {
        *modelIDs = vector<vector<int>>(modelLists.size());
        auto mIDs = (*modelIDs).begin();
        for (auto& mList : modelLists)
        {
            rawPredictions.push_back(mList.predict(x, &(*mIDs)));
            ++mIDs;
        }
    }
    else
    {
        for (auto& mList : modelLists)
        {
            rawPredictions.push_back(mList.predict(x));
        }
    }

    vector<vector<vector<double>>> result(nbPoints, vector<vector<double>>(nbOutputs));

    for (int i = 0; i < nbOutputs; ++i)
    {
        for (int k = 0; k < nbPoints; ++k)
        {
            result[k][i] = rawPredictions[i][k];
        }

    }

    return result;
}


template <typename ApproximatorType>
vector<vector<vector<double>>> Learner<ApproximatorType>::predict(vector<vector<double>> const& x, vector<vector<int>> *modelIDs)
{
    return predict(x, true, modelIDs);
}


// Dumps data in a file
template <typename ApproximatorType>
void Learner<ApproximatorType>::dumpEstimates(string fileName,
                                              vector<vector<double>> const& x,
                                              vector<vector<vector<double>>> const& estimates,
                                              vector<vector<vector<double>>> const& actual,
                                              vector<vector<int>> const& modelIDs)
{
    ofstream outputFile(fileName);
    if (!outputFile.is_open())
    {
        cerr << "Failed to open file" << endl;
        exit(1);
    }


    int const nbInputs = inputNames.size();
    int const nbDataPoints = x.size();
    bool const knownActual = (actual.size() > 0);
    bool const dumpModelIDs = (modelIDs.size() > 0);

    outputFile << inputNames[0];
    for (int i = 1; i < nbInputs; ++i)
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
        for (int i = 0; i < nbOutputs; ++i)
        {
            outputFile << ", output" << i << "_mID";
        }
    }

    outputFile << endl;

    for (int i = 0; i < nbDataPoints; ++i)
    {
        outputFile << x[i][0];
        for (int j = 1; j < nbInputs; ++j)
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


template <typename ApproximatorType>
void Learner<ApproximatorType>::predictAndDump(vector<vector<double>> const& x, string fileName, bool dumpModelIDs)
{
    if (dumpModelIDs)
    {
        vector<vector<int>> modelIDs;
        auto const estimates = predict(x, &modelIDs);
        Learner::dumpEstimates(fileName, x, estimates, vector<vector<vector<double>>>(0), &modelIDs);
    }
    else
    {
        Learner::dumpEstimates(fileName, x, predict(x));
    }
}


template <typename ApproximatorType>
void Learner<ApproximatorType>::predictAndDump(vector<vector<double>> const& x,
                                               vector<vector<vector<double>>> const& actual,
                                               string fileName,
                                               bool dumpModelIDs)
{
    if (dumpModelIDs)
    {
        vector<vector<int>> modelIDs;
        auto const estimates = predict(x, modelIDs);
        Learner<ApproximatorType>::dumpEstimates(fileName, x, estimates, actual, modelIDs);
    }
    else
    {
        Learner<ApproximatorType>::dumpEstimates(fileName, x, predict(x), actual);
    }
}



}
