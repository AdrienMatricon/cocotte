#ifndef COCOTTE_MODELLIST_H
#define COCOTTE_MODELLIST_H


#include <list>
#include <utility>
#include <string>
#include <boost/shared_ptr.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <opencv2/ml/ml.hpp>
#include <cocotte/models/models.hh>



namespace Cocotte {



template <typename ApproximatorType>
class ModelList final
{

private:

    static ApproximatorType approximator;

    // Each model is a binary tree :
    // - each leaf contains a point
    // - each node contains a form of the approximator that explains all points under it
    std::list<boost::shared_ptr<Models::Model>> models;

    size_t outputID;            // ID of the output in the DataPoint
    size_t nbInputDims;         // number of dimensions in the input
    size_t nbOutputDims;        // number of dimensions in the output
    size_t nbModels = 0;

    cv::RandomTrees classifier;
    bool trainedClassifier = false;

    // Comparison function to sort models by distance
    template <typename T>
    static bool pairCompareFirst(std::pair<double, T> const& lhs,
                            std::pair<double, T> const& rhs)
    {
        return std::get<0>(lhs) < std::get<0>(rhs);
    }

    // Utility function
    boost::shared_ptr<Models::Model> createLeaf(boost::shared_ptr<DataPoint const> point, bool markAsTemporary = false);


public:

    //Constructor
    ModelList() = default;
    explicit ModelList(int ouputID, int nbInputDims, int nbOutputDims);

    // Accessors
    size_t getNbModels() const;
    size_t getComplexity(size_t j) const;


    // Adding and removing models
    void addModel(boost::shared_ptr<Models::Model> model);
    boost::shared_ptr<Models::Model> firstModel();
    void removeFirstModel();

    // Creates leaves for the new points and merges them with models or submodels,
    // starting with the closest ones. We expect to get the same result
    // when adding points one by one, in batches, or all at once.
    void addPoint(boost::shared_ptr<DataPoint const> pointAddress);
    void addPoints(std::vector<boost::shared_ptr<DataPoint const>> const& pointAddresses);

    // Creates a leaf for a new point and tries to merge it with an already existing model,
    // Models created in this way are marked as temporary
    // Returns true if it succeeded, false if a new model was created
    bool tryAddingPointToExistingModels(boost::shared_ptr<DataPoint const> pointAddress);

    // Removes temporary models and add all points in temporary leaves with addPoints()
    void restructureModels();

    // Predict the value of a point
    void trainClassifier();
    std::vector<int> selectModels(std::vector<std::vector<double>> const& points);
    std::vector<std::vector<double>> predict(std::vector<std::vector<double>> const& points,
                                             bool fillModelIDs = false,
                                             std::vector<int> *modelIDs = nullptr);
    std::vector<std::vector<double>> predict(std::vector<std::vector<double>> const& points, std::vector<int> *modelIDs);

    // Returns a string that details the forms in the list
    std::string toString(std::vector<std::string> inputNames, std::vector<std::string> outputNames) const;

private:

    // Tries to merge two models into one without increasing complexity
    // Returns the result if it succeeded, and a default-constructed shared_ptr otherwise
    boost::shared_ptr<Models::Model> tryMerge(boost::shared_ptr<Models::Model> model0, boost::shared_ptr<Models::Model> model1, bool markAsTemporary = false);

    // Merges the models with each other, starting with the closest ones:
    // - atomicModels is a list of leaves, or more generally of models that are supposed correctly merged
    // - independentlyMergedModels is a list of models resulting from previous merges
    //   => Those merges may be rolled back because of the new models in atomicModels.
    //      We expect to get the same result with independentlyMergedModels or with the
    //      list of every leaf in independentlyMergedModels
    std::list<boost::shared_ptr<Models::Model>> mergeAsMuchAsPossible(std::list<boost::shared_ptr<Models::Model>>&& atomicModels,
                                                                      std::list<boost::shared_ptr<Models::Model>>&& independentlyMergedModels,
                                                                      bool markMergesAsTemporary = false);


    // Serialization
    template<typename Archive>
    friend void serialize(Archive& archive, ModelList<ApproximatorType>& mList, const unsigned int version)
    {
        archive.template register_type<Models::Leaf>();
        archive.template register_type<Models::Node>();
        archive & mList.models;
        archive & mList.outputID;
        archive & mList.nbInputDims;
        archive & mList.nbOutputDims;
        archive & mList.nbModels;
    }
};



}

#include <cocotte/modellist.hpp>

#endif
