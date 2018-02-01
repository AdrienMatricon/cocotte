#ifndef COCOTTE_MODELLIST_H
#define COCOTTE_MODELLIST_H


#include <list>
#include <utility>
#include <string>
#include <memory>
#include <boost/serialization/list.hpp>
//#include <boost/serialization/shared_ptr.hpp>
#include <opencv2/ml/ml.hpp>
#include <cocotte/models/models.hh>



namespace Cocotte {



template <typename ApproximatorType>
class ModelList final
{

private:

    // Each model is a binary tree :
    // - each leaf contains a point
    // - each node contains a form of the approximator that explains all points under it
    std::list<std::shared_ptr<Models::Model<ApproximatorType>>> models;

    unsigned int outputID;            // ID of the output in the DataPoint
    unsigned int nbInputDims;         // number of dimensions in the input
    unsigned int nbOutputDims;        // number of dimensions in the output
    unsigned int nbModels = 0;

    std::shared_ptr<cv::RandomTrees> classifier;


public:

    //Constructor
    ModelList() = default;
    explicit ModelList(unsigned int ouputID, unsigned int nbInputDims, unsigned int nbOutputDims);

    // Accessors
    unsigned int getNbModels() const;
    unsigned int getComplexity(unsigned int j) const;


    // Adding and removing models
    void addModel(std::shared_ptr<Models::Model<ApproximatorType>> model);
    std::shared_ptr<Models::Model<ApproximatorType>> firstModel();
    void removeFirstModel();

    // Creates leaves for the new points and merges them with models or submodels,
    // starting with the closest ones. We expect to get the same result
    // when adding points one by one, in batches, or all at once,
    // except if noRollback (previous merges are kept)
    // or addToExistingModelsOnly (new leaves merged into old models first) is set to true.
    // In that case, merging goes faster and new leaves/nodes are marked as temporary
    void addPoint(std::shared_ptr<DataPoint const> pointAddress, bool noRollback = false);
    void addPoints(std::vector<std::shared_ptr<DataPoint const>> const& pointAddresses,
                   bool noRollback = false,
                   bool addToExistingModelsOnly = false);

    // Removes temporary models and add all points in temporary leaves with addPoints()
    // (with noRollback and addToExistingModelsOnly set to false)
    void restructureModels();

    // Checks if for each point there exists a model that could predict it
    bool canBePredicted(std::vector<std::shared_ptr<DataPoint const>> const& pointAddresses);

    // Predict the value of a point
    void trainClassifier();
    std::vector<unsigned int> selectModels(std::vector<std::vector<double>> const& points);
    std::vector<std::vector<double>> predict(std::vector<std::vector<double>> const& points,
                                             bool fillModelIDs = false,
                                             std::vector<unsigned int> *modelIDs = nullptr);
    std::vector<std::vector<double>> predict(std::vector<std::vector<double>> const& points, std::vector<unsigned int> *modelIDs);

    // Returns a string that details the forms in the list
    std::string toString(std::vector<std::string> inputNames, std::vector<std::string> outputNames) const;

private:

    // Comparison function to sort models by distance
    template <typename T, typename U>
    static bool pairCompareFirst(std::pair<T, U> const& lhs,
                                 std::pair<T, U> const& rhs)
    {
        return std::get<0>(lhs) < std::get<0>(rhs);
    }

    template <typename T>
    struct HasGreaterDistance
    {
        bool operator()(std::pair<Models::ModelDistance, T> const& lhs,
                        std::pair<Models::ModelDistance, T> const& rhs)
        {
            return !pairCompareFirst(lhs,rhs);
        }
    };

    // Utility function
    std::shared_ptr<Models::Model<ApproximatorType>> createLeaf(std::shared_ptr<DataPoint const> point, bool markAsTemporary = false);

    // Tries to merge models into one without increasing complexity
    // Returns the result if it succeeded, and a default-constructed shared_ptr otherwise
    std::shared_ptr<Models::Model<ApproximatorType>> tryMerge(
            std::vector<std::shared_ptr<Models::Model<ApproximatorType>>> candidateModels,
            bool markAsTemporary = false);

    // Merges the models with each other, starting with the closest ones:
    // - atomicModels is a list of leaves, or more generally of models that are supposed correctly merged
    // - independentlyMergedModels is a list of models resulting from previous merges
    //   => Those merges may be rolled back because of the new models in atomicModels.
    //      We expect to get the same result with independentlyMergedModels or with the
    //      list of every leaf in independentlyMergedModels
    // - if addingToExistingModelsis set to true:
    //   => atomicModels will not be merged with each other before being merged with those in independentlyMergedModels
    //   => no rollback will be done
    std::list<std::shared_ptr<Models::Model<ApproximatorType>>> mergeAsMuchAsPossible(
            std::list<std::shared_ptr<Models::Model<ApproximatorType>>>&& atomicModels,
            std::list<std::shared_ptr<Models::Model<ApproximatorType>>>&& independentlyMergedModels,
            bool noRollback = false,
            bool addToExistingModelsOnly = false);


    // Serialization
    template<typename Archive>
    friend void serialize(Archive& archive, ModelList<ApproximatorType>& mList, const unsigned int version)
    {
        (void) version; // Unused parameter

        archive.template register_type<Models::Leaf<ApproximatorType>>();
        archive.template register_type<Models::Node<ApproximatorType>>();
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
