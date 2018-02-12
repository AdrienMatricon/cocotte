
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
Learner<ApproximatorType>::Learner(std::string fileName)
{
    using std::ifstream;
    using std::cerr;
    using std::endl;
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
void Learner<ApproximatorType>::dumpModels(std::string fileName)
{
    using std::ofstream;
    using std::cerr;
    using std::endl;
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
void Learner<ApproximatorType>::addDataPoints(std::vector<DataPoint> const& points)
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
