
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <memory>
#include<boost/archive/text_iarchive.hpp>
#include<boost/archive/text_oarchive.hpp>
#include <cocotte/datatypes.h>
#include <cocotte/models/models.hh>
#include <cocotte/modellist.h>
#include <cocotte/learner.h>

namespace Cocotte {



// Constructor
template <typename ApproximatorType>
Learner<ApproximatorType>::Learner(std::vector<std::string> const& iNames,
                                   std::vector<std::vector<std::string>> const& oNames):
    inputNames(iNames), outputNames(oNames), nbOutputs(oNames.size())
{
    unsigned int const nbInputDims = iNames.size();
    modelLists.reserve(oNames.size());
    for (unsigned int i = 0; i < nbOutputs; ++i)
    {
        modelLists.push_back(ModelList<ApproximatorType>(i, nbInputDims, oNames[i].size()));
    }
}



// Accessors
template <typename ApproximatorType>
std::vector<std::string> Learner<ApproximatorType>::getInputNames() const
{
    return inputNames;
}


template <typename ApproximatorType>
std::vector<std::vector<std::string>> Learner<ApproximatorType>::getOutputNames() const
{
    return outputNames;
}


template <typename ApproximatorType>
unsigned int Learner<ApproximatorType>::getNbPoints() const
{
    return data.size();
}


template <typename ApproximatorType>
unsigned int Learner<ApproximatorType>::getComplexity(unsigned int i, unsigned int j) const
{
    return modelLists[i].getComplexity(j);
}



// Serialization
template <typename ApproximatorType>
Learner<ApproximatorType>::Learner(string fileName)
{
    using std::ifstream;
    using boost::archive::text_iarchive;

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
    using std::ofstream;
    using boost::archive::text_oarchive;

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
    using std::shared_ptr;

    data.push_back(shared_ptr<DataPoint const>(new DataPoint(point)));

    for (auto& mList : modelLists)
    {
        mList.addPoint(data.back());
    }
}


template <typename ApproximatorType>
void Learner<ApproximatorType>::addDataPoints(vector<DataPoint> const& points)
{
    using std::vector;
    using std::shared_ptr;

    data.reserve(data.size() + points.size());
    vector<shared_ptr<DataPoint const>> pointers;

    for (auto const& point : points)
    {
        data.push_back(shared_ptr<DataPoint const>(new DataPoint(point)));
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
    using std::move;
    using std::vector;
    using std::shared_ptr;

    for (auto& mList: modelLists)
    {
        // For each old model, we check if all its points could all fit in other models
        unsigned int nbModels = mList.getNbModels();
        bool keepGoing = true;

        while (keepGoing && (nbModels >= 2))
        {
            keepGoing = false;

            for (unsigned int j = 0; j < nbModels; ++j)
            {
                // We remove the first model
                auto firstModel = mList.firstModel();
                mList.removeFirstModel();

                // We save the information we need to roll back what we will do
                auto const oldList = mList;

                // We build a vector of all datapoints in the first model
                vector<shared_ptr<DataPoint const>> pointers;
                pointers.reserve(firstModel->getNbPoints());

                for (auto mIt = Models::pointsBegin(firstModel), mEnd = Models::pointsEnd(firstModel);
                     mIt != mEnd; ++mIt)
                {
                    pointers.push_back(mIt.getSharedPointer());
                }

                // We try to distribute the points into other models
                mList.addPoints(pointers, true, true);

                unsigned int newNbModels = mList.getNbModels();
                if (newNbModels < nbModels)
                {
                    // If it worked, we keep removing artifacts
                    keepGoing = true;
                    nbModels = newNbModels;
                    continue;
                }
                else
                {
                    // If it did not work, we try to distribute other models
                    // and stop if no model could be distributed
                    mList = move(oldList);
                    mList.addModel(firstModel);
                }
            }
        }
    }
}


// Faster, greedy, the order in which the points are added matters
template <typename ApproximatorType>
void Learner<ApproximatorType>::addDataPointNoRollback(DataPoint const& point)
{
    using std::shared_ptr;

    data.push_back(shared_ptr<DataPoint const>(new DataPoint(point)));

    for (auto& mList : modelLists)
    {
        mList.addPoint(data.back(), true);
    }
}


template <typename ApproximatorType>
void Learner<ApproximatorType>::addDataPointsNoRollback(std::vector<DataPoint> const& points)
{
    using std::vector;
    using std::shared_ptr;

    data.reserve(data.size() + points.size());
    vector<shared_ptr<DataPoint const>> pointers;

    for (auto const& point : points)
    {
        data.push_back(shared_ptr<DataPoint const>(new DataPoint(point)));
        pointers.push_back(data.back());
    }

    for (auto& mList : modelLists)
    {
        mList.addPoints(pointers, true);
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
}


// For fully incremental learning
// Rolls back removeArtifacts(), adds the points,
// Then calls removeArtifacts()
template <typename ApproximatorType>
void Learner<ApproximatorType>::addDataPointIncremental(DataPoint const& point)
{
    using std::shared_ptr;

    data.push_back(shared_ptr<DataPoint const>(new DataPoint(point)));

    for (auto& mList : modelLists)
    {
        mList.restructureModels();
        mList.addPoint(data.back());
    }

    removeArtifacts();
}


template <typename ApproximatorType>
void Learner<ApproximatorType>::addDataPointsIncremental(std::vector<DataPoint> const& points)
{
    using std::vector;
    using std::shared_ptr;

    data.reserve(data.size() + points.size());
    vector<shared_ptr<DataPoint const>> pointers;

    for (auto const& point : points)
    {
        data.push_back(shared_ptr<DataPoint const>(new DataPoint(point)));
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
std::vector<std::vector<std::vector<double>>> Learner<ApproximatorType>::predict(std::vector<std::vector<double>> const& x,
                                                                                 bool shouldObtainModelIDs,
                                                                                 std::vector<std::vector<unsigned int>> *modelIDs)
{
    using std::vector;

    unsigned int const nbPoints = x.size();
    vector<vector<vector<double>>> rawPredictions;
    rawPredictions.reserve(nbOutputs);

    if (shouldObtainModelIDs)
    {
        *modelIDs = vector<vector<unsigned int>>(modelLists.size());
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

    for (unsigned int i = 0; i < nbOutputs; ++i)
    {
        for (unsigned int k = 0; k < nbPoints; ++k)
        {
            result[k][i] = rawPredictions[i][k];
        }

    }

    return result;
}


template <typename ApproximatorType>
std::vector<std::vector<std::vector<double>>> Learner<ApproximatorType>::predict(std::vector<std::vector<double>> const& x,
                                                                                 std::vector<std::vector<unsigned int>> *modelIDs)
{
    return predict(x, true, modelIDs);
}



}
