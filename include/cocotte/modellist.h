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

    int outputID;           // ID of the output in the DataPoint
    int nbInputDims;        // number of dimensions in the input
    int nbOutputDims;       // number of dimensions in the output
    int nbModels = 0;

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

    // Tries to merge two models into one without increasing complexity
    // Returns the result if it succeeded, and a default-constructed shared_ptr otherwise
    boost::shared_ptr<Models::Model> tryMerge(boost::shared_ptr<Models::Model> model0, boost::shared_ptr<Models::Model> model1, bool markAsTemporary = false);

    // Takes a new leaf L as well as a pair (distance,iterator) to the closest leaf CL in a given existing model M
    // This function searches for the smallest submodel S containing CL of M such that:
    //  1) distance(L,S) <= distance(L, M\S)
    //  2) distance(L,S) <= distance(S, M\S)
    // then tries to merge L and S and returns if it succeeded
    // In case of success, the new model is added to the model list,
    // all predecessors of S are destroyed, and the other branches (submodels) are added to the model list
    bool tryToInsertLeafIn(boost::shared_ptr<Models::Model> leaf, std::pair<double, Models::ModelIterator> distAndBranch);

    // Tries all merges between models in the model list, starting with the closest ones
    void doAllPossibleMerges();

    // Creates leaves for the new points, tries to add them to existing models
    // with tryToInsertLeafIn(), and creates new models if necessary
    void addPoint(boost::shared_ptr<DataPoint const> pointAddress);
    void addPoints(std::vector<boost::shared_ptr<DataPoint const>> const& pointAddresses);

    // Creates a leaf for a new point and tries to merge it with an already existing model,
    // without care for the notion of proximity maintained by tryToInsertLeafIn()
    // Models so created are marked as temporary
    // Returns true if it succeeded, false if a new model was created
    bool tryAddingPointToExistingModels(boost::shared_ptr<DataPoint const> pointAddress);

    // Removes temporary models, adds the submodels to the list then calls doAllPossibleMerges()
    // Restores the notion of proximity maintained by tryToInsertLeafIn()
    void restructureModels();

    // Predict the value of a point
    void trainClassifier();
    std::vector<int> selectModels(std::vector<std::vector<double>> const& points);
    std::vector<std::vector<double>> predict(std::vector<std::vector<double>> const& points,
                                             bool dumpModelIDs = false,
                                             std::vector<int> *modelIDs = nullptr);
    std::vector<std::vector<double>> predict(std::vector<std::vector<double>> const& points, std::vector<int> *modelIDs);

    // Returns a string that details the forms in the list
    std::string toString(std::vector<std::string> inputNames, std::vector<std::string> outputNames) const;

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
