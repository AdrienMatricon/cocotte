#ifndef COCOTTE_LEARNER_H
#define COCOTTE_LEARNER_H


#include <vector>
#include <string>
#include <ostream>
#include <boost/shared_ptr.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <cocotte/datatypes.h>
#include <cocotte/modellist.h>



namespace Cocotte {



template <typename ApproximatorType>
class Learner final
{

private:

    std::vector<std::string> inputNames;
    std::vector<std::vector<std::string>> outputNames;
    int nbOutputs;
    std::vector<ModelList<ApproximatorType>> modelLists;
    std::vector<boost::shared_ptr<DataPoint const>> data;

public:

    // Constructor
    Learner(std::vector<std::string> const& inputNames, std::vector<std::vector<std::string>> const& outputNames);

    // Accessors
    std::vector<std::string> getInputNames() const;
    std::vector<std::vector<std::string>> getOutputNames() const;
    size_t getNbPoints() const;
    size_t getComplexity(size_t i, size_t j) const;

    // Serialization
    explicit Learner(std::string fileName);
    void dumpModels(std::string fileName);


    // Main methods

    // Points can be added in any order
    void addDataPoint(DataPoint const& point);
    void addDataPoints(std::vector<DataPoint> const& points);

    // Should be called after adding all points
    void removeArtifacts();

    // Faster, greedy, the order in which the points are added matters
    void addDataPointToExistingModels(DataPoint const& point);
    void addDataPointsToExistingModels(std::vector<DataPoint> const& points);

    // Makes it as if all points were added with addDataPoints(), then runs removeArtifacts()
    void restructureModels();

    // For fully incremental learning
    // Rolls back removeArtifacts(), adds the points,
    // Then calls removeArtifacts()
    void addDataPointIncremental(DataPoint const& point);
    void addDataPointsIncremental(std::vector<DataPoint> const& points);

    // Predicts outputs for new points
    std::vector<std::vector<std::vector<double>>> predict(std::vector<std::vector<double>> const& x,
                                                          bool shouldGetModelIDs = false,
                                                          std::vector<std::vector<int>> *modelIDs = nullptr);
    std::vector<std::vector<std::vector<double>>> predict(std::vector<std::vector<double>> const& x, std::vector<std::vector<int>> *modelIDs);


    template<typename Archive>
    friend void serialize(Archive& archive, Learner<ApproximatorType>& learner, const unsigned int version)
    {
        archive & learner.inputNames;
        archive & learner.outputNames;
        archive & learner.nbOutputs;
        archive & learner.modelLists;
        archive & learner.data;
    }

    friend std::ostream& operator<< (std::ostream& out, Learner<ApproximatorType> const& learner)
    {
        out << "Model for output 0:" << endl;
        out << learner.modelLists[0].toString(learner.inputNames, learner.outputNames[0]);

        for (int i = 1; i < learner.nbOutputs; ++i)
        {
            out << endl;
            out << "Models for output " << i << ":" << endl;
            out << learner.modelLists[i].toString(learner.inputNames, learner.outputNames[i]);
        }

        return out;
    }
};



}



#include <cocotte/learner.hpp>

#endif
