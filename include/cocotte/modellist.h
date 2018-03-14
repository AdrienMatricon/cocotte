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
#include <cocotte/useddimensions.h>



namespace Cocotte {



template <typename ApproximatorType>
class ModelList final
{

private:

    // Each model is a tree :
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
    void removeModel(std::shared_ptr<Models::Model<ApproximatorType>> model);
    std::shared_ptr<Models::Model<ApproximatorType>> firstModel();
    void removeFirstModel();

    // Creates leaves for the new points and merges them with models or submodels,
    // starting with the closest ones
    void addPoint(std::shared_ptr<DataPoint const> pointAddress);
    void addPoints(std::vector<std::shared_ptr<DataPoint const>> const& pointAddresses);

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
    std::shared_ptr<Models::Model<ApproximatorType>> createLeaf(std::shared_ptr<DataPoint const> point);

    // - Tries to merge models into one without increasing complexity
    // - Returns the result if it succeeded, and a default-constructed shared_ptr otherwise
    // - shouldWork can be set to specify complexities and dimensions that should allow fitting,
    //    to make sure that a solution will be found
    std::shared_ptr<Models::Model<ApproximatorType>> tryMerge(
            std::vector<std::shared_ptr<Models::Model<ApproximatorType>>> candidateModels,
            std::vector<std::pair<unsigned int, UsedDimensions>> const& shouldWork= {});

    // Tries have one model "steal" points (or rather, submodels) from each other:
    // - candidate0 and candidate1 are models containing only one point (leaves)
    // - the top-level model containing candidate0 tries to steal candidate1
    //    or a model that contains it, and vice-versa
    // - we go with the option which leads to the lowest sum of complexities
    // => if point stealing was possible: we return true and the new models
    //    otherwise: we return false and an empty list
    std::pair<bool, std::list<std::shared_ptr<Models::Model<ApproximatorType>>>> pointStealing(
            Models::ModelIterator<ApproximatorType> candidate0,
            Models::ModelIterator<ApproximatorType> candidate1);

    // Merge models as much as possible by having them "steal" points (or rather, submodels) from each other,
    //  starting with the closest pair of submodels
    void performPointStealing();


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
