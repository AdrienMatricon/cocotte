
#include <cmath>
using std::min;
#include <list>
using std::list;
#include <vector>
using std::vector;
#include <utility>
using std::pair;
#include <boost/shared_ptr.hpp>
using boost::shared_ptr;
using boost::static_pointer_cast;
using boost::const_pointer_cast;
#include <algorithm>
using std::sort;
#include <string>
using std::string;
#include <sstream>
using std::stringstream;
#include <iostream>
using std::endl;
#include <cocotte/approximators/approximator.h>
using Cocotte::Approximators::Form;
#include <cocotte/models/models.hh>
using Cocotte::Models::Model;
using Cocotte::Models::Leaf;
using Cocotte::Models::Node;
using Cocotte::Models::ModelIterator;
using Cocotte::Models::ModelConstIterator;
#include <cocotte/modellist.h>

namespace Cocotte {



template <typename ApproximatorType>
ApproximatorType ModelList<ApproximatorType>::approximator;



// Utility function
template <typename ApproximatorType>
shared_ptr<Model> ModelList<ApproximatorType>::createLeaf(boost::shared_ptr<DataPoint const> point, bool markAsTemporary)
{
    vector<Form> forms;
    forms.reserve(nbOutputDims);

    for (auto const outDim : point->t[outputID])
    {
        forms.push_back(approximator.fitOnePoint(outDim.value, nbInputDims));
    }

    return shared_ptr<Model>(new Leaf(forms, point, markAsTemporary));
}



// Constructor
template <typename ApproximatorType>
ModelList<ApproximatorType>::ModelList(int oID, int nId, int nOd): outputID(oID), nbInputDims(nId), nbOutputDims(nOd)
{}



// Accessors
template <typename ApproximatorType>
size_t ModelList<ApproximatorType>::getNbModels() const
{
    return nbModels;
}


template <typename ApproximatorType>
size_t ModelList<ApproximatorType>::getComplexity(size_t j) const
{
    size_t complexity = 0;

    for (auto const& model : models)
    {
        complexity += model->getForms()[j].complexity;
    }

    return complexity;
}



// Adding and removing models
template <typename ApproximatorType>
void ModelList<ApproximatorType>::addModel(shared_ptr<Model> model)
{
    models.push_back(model);
    ++nbModels;
    trainedClassifier = false;
}


template <typename ApproximatorType>
shared_ptr<Model> ModelList<ApproximatorType>::firstModel()
{
    return models.front();
}


template <typename ApproximatorType>
void ModelList<ApproximatorType>::removeFirstModel()
{
    models.pop_front();
    --nbModels;
    trainedClassifier = false;
}



// Tries to merge two models into one without increasing complexity
// Returns the result if it succeeded, and a default-constructed shared_ptr otherwise
template <typename ApproximatorType>
shared_ptr<Model> ModelList<ApproximatorType>::tryMerge(shared_ptr<Model> model0, shared_ptr<Model> model1, bool markAsTemporary)
{
    // We determine if new node should be temporary
    markAsTemporary = (markAsTemporary || model0->isTemporary() || model1->isTemporary());

    shared_ptr<Model> node (new Node(model0, model1, markAsTemporary));
    int const nbPoints = node->getNbPoints();
    auto const mBegin = Models::pointsBegin(const_pointer_cast<Model const>(node));
    auto const mEnd = Models::pointsEnd(const_pointer_cast<Model const>(node));

    vector<Form> forms0 = model0->getForms(), forms1 = model1->getForms();
    vector<Form> newForms;

    for (int dim = 0; dim < nbOutputDims; ++dim)
    {
        Form const form0 = forms0[dim], form1 = forms1[dim];
        UsedDimensions availableDimensions = form0.usedDimensions + form1.usedDimensions;

        bool success = false;
        Form newForm;

        // We check that a merge is possible
        {
            // We list the most complex forms available
            list<list<Form>> possibleForms = approximator.getMostComplexForms(availableDimensions, form0.complexity + form1.complexity);

            for (auto& someForms : possibleForms)
            {
                for (auto& form : someForms)
                {
                    // For each form, we try to fit all points
                    if (approximator.tryFit(form, nbPoints, mBegin, mEnd, outputID, dim))
                    {
                        success = true;
                        newForm = form;
                        break;
                    }
                }

                if (success)
                {
                    break;
                }
            }
        }

        if (!success)
        {
            // We did not succeed, so we return a default-constructed shared_ptr
            return shared_ptr<Model>();
        }

        // Otherwise, if we succeeded, we try to find the best form
        int minSuccess = newForm.complexity;
        int maxFail = min(form0.complexity, form1.complexity) - 1;  // Sure to fail

        while (maxFail != minSuccess - 1)
        {
            int middle = (maxFail + minSuccess) / 2;
            list<list<Form>> possibleForms = approximator.getMostComplexForms(availableDimensions, middle);

            success = false;

            for (auto& someForms : possibleForms)
            {
                for (auto& form : someForms)
                {
                    // For each form, we try to fit all points
                    if (approximator.tryFit(form, nbPoints, mBegin, mEnd, outputID, dim))
                    {
                        success = true;
                        newForm = form;
                        minSuccess = newForm.complexity;
                        break;
                    }
                }

                if (success)
                {
                    break;
                }
            }

            if (!success)
            {
                maxFail = middle;
            }
        }

        newForms.push_back(newForm);
    }

    static_pointer_cast<Node>(node)->setForms(newForms);
    return node;
}



// Takes a new leaf L as well as a pair (distance,iterator) to the closest leaf CL in a given existing model M
// This function searches for the smallest submodel S containing CL of M such that:
//  1) distance(L,S) <= distance(L, M\S)
//  2) distance(L,S) <= distance(S, M\S)
// then tries to merge L and S and returns if it succeeded
// In case of success, the new model is added to the model list,
// all predecessors of S are destroyed, and the other branches (submodels) are added to the model list
template <typename ApproximatorType>
bool ModelList<ApproximatorType>::tryToInsertLeafIn(shared_ptr<Model> leaf, pair<double, ModelIterator> distAndBranch)
{
    double const dist = std::get<0>(distAndBranch);
    list<shared_ptr<Model>> treeBranch = std::get<1>(distAndBranch).getTreeBranch();
    shared_ptr<Model> closest = treeBranch.back();  // Closest Leaf to L in M
    treeBranch.pop_back();

    // We go up the tree - keeping 1) verified - until 2) is verified or S is M
    if (!treeBranch.empty())
    {
        auto predecessor = static_pointer_cast<Node>(treeBranch.back());
        // Because of how the tree was made, distance(S, M\S) can be computed locally
        while(Models::getDistance(predecessor->getModel0(), predecessor->getModel1(), outputID) < dist)
        {
            closest = treeBranch.back();
            treeBranch.pop_back();
            if (treeBranch.empty())
            {
                // S is M
                break;
            }

            predecessor = static_pointer_cast<Node>(treeBranch.back());
        }
    }

    // Then we try merging
    shared_ptr<Model> newModel = tryMerge(closest, leaf);

    // If the merge fails, the insertion did not succeed
    // Otherwise, we need to insert things in the model list
    if (!newModel)
    {
        return false;
    }

    if (treeBranch.empty())
    {
        // If S is M, we replace the old model by the new one
        for (auto& model:models)
        {
            if (model.get() == closest.get())
            {
                model = newModel;
            }
        }
    }
    else
    {
        // Otherwise, we also replace the old model by the new one
        {
            auto const top = treeBranch.front();
            for (auto& model:models)
            {
                if (model.get() == top.get())
                {
                    model = newModel;
                }
            }
        }

        // But we add all orphan submodels in the model list as well
        {
            auto currentPointer = closest.get();
            auto predecessor = static_pointer_cast<Node>(treeBranch.back());
            auto model0 = predecessor->getModel0();

            treeBranch.pop_back();

            if (model0.get() == currentPointer)
            {
                addModel(predecessor->getModel1());
            }
            else
            {
                addModel(model0);
            }

            while (!treeBranch.empty())
            {
                currentPointer = predecessor.get();
                predecessor = static_pointer_cast<Node>(treeBranch.back());
                model0 = predecessor->getModel0();

                treeBranch.pop_back();

                if (model0.get() == currentPointer)
                {
                    addModel(predecessor->getModel1());
                }
                else
                {
                    addModel(model0);
                }

            }
        }
    }

    return true;
}



// Tries all merges between models in the model list, starting with the closest ones
template <typename ApproximatorType>
void ModelList<ApproximatorType>::doAllPossibleMerges()
{
    using ModelPair = pair<shared_ptr<Model>, shared_ptr<Model>>;

    vector<pair<double, ModelPair>> distanceQueue;
    distanceQueue.reserve(nbModels*nbModels);

    // We compute the distance between all pairs of models
    for (auto const& model : models)
    {
        for (auto const& otherModel : models)
        {
            if (model.get() != otherModel.get())
            {
                distanceQueue.push_back(pair<double, ModelPair>(
                                            Models::getDistance(model, otherModel, outputID),
                                            ModelPair(model, otherModel)));
            }
        }
    }

    // We sort them out
    sort(distanceQueue.begin(), distanceQueue.end(), pairCompareFirst<ModelPair>);

    double const maxDist = std::get<0>(distanceQueue.back());   // biggest distance between two points
    double const supDist = maxDist*2 + 0.01;                    // huge distance to assign to pairs that need to be removed
    double const cutOff = maxDist*1.5 + 0.001;                  // cutoff between the two


    // Starting with the closest pair, we try merging models
    bool keepMerging = true;

    while (keepMerging)
    {
        keepMerging = false;

        for (auto& current : distanceQueue)
        {
            auto const& mPair = std::get<1>(current);
            auto const& first = std::get<0>(mPair);
            auto const& second = std::get<1>(mPair);

            auto newModel = tryMerge(first, second);

            if (newModel)
            {
                bool firstOccurrence = true;
                // We insert the new model and remove the old ones
                for (auto mIt = models.begin(), mEnd = models.end();
                     mIt != mEnd; ++mIt)
                {
                    if ( (mIt->get() == first.get()) || (mIt->get() == second.get()))
                    {
                        if (firstOccurrence)
                        {
                            *mIt = newModel;
                            firstOccurrence = false;
                        }
                        else
                        {
                            models.erase(mIt);
                            --nbModels;
                            break;
                        }
                    }
                }

                // To remove all pairs of candidate models including one that was just merged,
                // we assign them to a high distance (above cutoff), sort the distance queue then truncate it
                for (auto& d : distanceQueue)
                {
                    auto const& otherPair = std::get<1>(d);
                    auto const& otherFirst = std::get<0>(otherPair);
                    auto const& otherSecond = std::get<1>(otherPair);

                    if ( (first.get() == otherFirst.get()) || (first.get() == otherSecond.get())
                         || (second.get() == otherFirst.get()) || (second.get() == otherSecond.get()))
                    {
                        std::get<0>(d) = supDist;
                    }
                }

                // Before sorting, we compute the distances from all models to the new one
                // and add them to the distance queue
                for (auto const& model : models)
                {
                    if (model.get() != newModel.get())
                    {
                        distanceQueue.push_back(pair<double, ModelPair>(
                                                    Models::getDistance(model, newModel, outputID),
                                                    ModelPair(model, newModel)));
                    }
                }

                // We sort the distance queue and remove element over the cutOff
                sort(distanceQueue.begin(), distanceQueue.end(), pairCompareFirst<ModelPair>);
                for (auto dIt = distanceQueue.begin(), dEnd = distanceQueue.end();
                     dIt != dEnd; ++dIt)
                {
                    if (std::get<0>(*dIt) > cutOff)
                    {
                        distanceQueue.erase(dIt, dEnd);
                        break;
                    }
                }

                keepMerging = true;
                break;  // we modified the distance queue, so we go back to the closest pair
            }
            else
            {
                // If the merge failed, we assign the pair to a high distance (above cutoff),
                // so that it is removed the next time the queue is sorted then truncated
                std::get<0>(current) = supDist;
            }
        }
    }
}



// Creates leaves for the new points, tries to add them to existing models
// with tryToInsertLeafIn(), and creates new models if necessary
template <typename ApproximatorType>
void ModelList<ApproximatorType>::addPoint(boost::shared_ptr<DataPoint const> pointAddress)
{
    addPoints(vector<boost::shared_ptr<DataPoint const>>(1,pointAddress));
}


template <typename ApproximatorType>
void ModelList<ApproximatorType>::addPoints(vector<boost::shared_ptr<DataPoint const>> const& pointAddresses)
{
    trainedClassifier = false;

    if (nbModels == 0)
    {
        for (auto const& pointAddress : pointAddresses)
        {
            shared_ptr<Model> newLeaf = createLeaf(pointAddress);
            addModel(newLeaf);
        }
    }
    else
    {
        for (auto const& pointAddress : pointAddresses)
        {
            shared_ptr<Model> newLeaf = createLeaf(pointAddress);

            // We compute the distances to all models
            vector<pair<double, ModelIterator>> distancesAndModels;
            distancesAndModels.reserve(nbModels);

            DataPoint const& point = *pointAddress;
            for (auto& model : models)
            {
                distancesAndModels.push_back(Models::getClosest(point, model, outputID));
            }

            // We look for the closest model that works
            sort(distancesAndModels.begin(), distancesAndModels.end(), pairCompareFirst<ModelIterator>);

            bool success = false;
            for (auto const& currentPair : distancesAndModels)
            {
                success = tryToInsertLeafIn(newLeaf, currentPair);
                if (success)
                {
                    break;
                }
            }

            if (!success)
            {
                addModel(newLeaf);
            }
        }
    }

    // Then we try all merges between models in the model list
    if (nbModels > 1)
    {
        doAllPossibleMerges();
    }
}



// Creates a leaf for a new point and tries to merge it with an already existing model,
// without care for the notion of proximity maintained by tryToInsertLeafIn()
// Models so created are marked as temporary
// Returns true if it succeeded, false if a new model was created
template <typename ApproximatorType>
bool ModelList<ApproximatorType>::tryAddingPointToExistingModels(boost::shared_ptr<DataPoint const> pointAddress)
{
    using ListIterator = list<boost::shared_ptr<Models::Model>>::iterator;

    trainedClassifier = false;
    shared_ptr<Model> newLeaf = createLeaf(pointAddress, true);

    if (nbModels > 0)
    {
        // We compute the distances to all models
        vector<pair<double, ListIterator>> distancesAndModels;
        distancesAndModels.reserve(nbModels);

        for (auto mIt = models.begin(), mEnd = models.end(); mIt != mEnd; ++mIt)
        {
            double const dist = Models::getDistance(newLeaf, *mIt, outputID);
            distancesAndModels.push_back(pair<double, ListIterator>(dist, mIt));
        }

        // We try the merge with the closest models first
        sort(distancesAndModels.begin(), distancesAndModels.end(), pairCompareFirst<ListIterator>);

        for (auto& currentPair : distancesAndModels)
        {
            auto newModel = tryMerge(newLeaf, *(currentPair.second), true);

            if (newModel)
            {
                *(currentPair.second) = newModel;
                doAllPossibleMerges();
                return true;
            }
        }
    }

    addModel(newLeaf);
    return false;
}



// Removes temporary models, adds the submodels to the list then calls doAllPossibleMerges()
// Restores the notion of proximity maintained by tryToInsertLeafIn()
template <typename ApproximatorType>
void ModelList<ApproximatorType>::restructureModels()
{
    trainedClassifier = false;

    list<shared_ptr<Model>> toProcess = std::move(models);
    models = list<shared_ptr<Model>>();
    nbModels = 0;

    vector<shared_ptr<DataPoint const>> pointsInTemporaryLeaves;

    while(!toProcess.empty())
    {
        auto current = toProcess.back();
        toProcess.pop_back();

        if (current->isTemporary())
        {
            if (current->isLeaf())
            {
                pointsInTemporaryLeaves.push_back(static_pointer_cast<Leaf>(current)->getPointAddress());
            }
            else
            {
                shared_ptr<Node> asNode = static_pointer_cast<Node>(current);
                toProcess.push_back(asNode->getModel0());
                toProcess.push_back(asNode->getModel1());
            }
        }
        else
        {
            addModel(current);
        }
    }

    addPoints(pointsInTemporaryLeaves);
}



// Predict the value of a point
template <typename ApproximatorType>
void ModelList<ApproximatorType>::trainClassifier()
{
    int const nbVars = Models::pointsBegin(models.front())->x.size();

    vector<float> priors(nbModels, 1.f);
    cv::RandomTreeParams params(nbModels*2,                         // max depth
                                2,                                  // min sample count
                                0.f,                                // regression accuracy: N/A here
                                false,                              // compute surrogate split, no missing data
                                15,                                 // max number of categories (use sub-optimal algorithm for larger numbers)
                                priors.data(),                      // the array of priors
                                false,                              // calculate variable importance
                                nbVars,                             // number of variables randomly selected at node and used to find the best split(s).
                                1000,                               // max number of trees in the forest
                                0.01f,                              // forest accuracy
                                CV_TERMCRIT_ITER |	CV_TERMCRIT_EPS // termination cirteria
                                );

    int nbPoints = 0;
    for (auto const& model : models)
    {
        nbPoints += model->getNbPoints();
    }


    cv::Mat data(nbPoints, nbVars, CV_32F);
    cv::Mat classification(nbPoints,1, CV_32F);

    {
        float label = 0.f;
        int i = 0;
        for (auto const& model : models)
        {
            auto const mEnd = Models::pointsEnd(model);
            for (auto mIt = Models::pointsBegin(model); mIt != mEnd; ++mIt, ++i)
            {
                auto const& x = mIt->x;
                for (int j = 0; j < nbVars; ++j)
                {
                    data.at<float>(i,j) = x[j].value;
                }

                classification.at<float>(i, 0) = label;
            }

            label += 1.0f;
        }
    }

    classifier.train(data, CV_ROW_SAMPLE, classification, cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat(), params);
    trainedClassifier = true;
}


template <typename ApproximatorType>
vector<int> ModelList<ApproximatorType>::selectModels(vector<vector<double>> const& points)
{
    size_t const nbPoints = points.size();
    size_t const nbVars = points[0].size();
    vector<int> result(nbPoints);

    if (!trainedClassifier)
    {
        trainClassifier();
    }

    for (size_t i = 0; i < nbPoints; ++i)
    {
        cv::Mat point(1, nbVars, CV_32F);
        for (size_t j = 0; j < nbVars; ++j)
        {
            point.at<float>(0,j) = points[i][j];
        }

        result[i] = static_cast<int>(classifier.predict(point) + 0.5f);
    }

    return result;
}


template <typename ApproximatorType>
vector<vector<double>> ModelList<ApproximatorType>::predict(vector<vector<double>> const& points, bool dumpModelIDs, vector<int> *modelIDs)
{
    size_t const nbPoints = points.size();
    vector<vector<vector<double>>> possibleValues;
    possibleValues.reserve(nbModels);

    for (auto const& model : models)
    {
        auto const forms = model->getForms();
        vector<vector<double>> estimates;
        estimates.reserve(nbOutputDims);
        for (auto const& f : forms)
        {
            estimates.push_back(approximator.estimate(f, points));
        }
        possibleValues.push_back(estimates);
    }

    vector<int> const selected = selectModels(points);
    if (dumpModelIDs)
    {
        *modelIDs = selected;
    }

    vector<vector<double>> t(nbPoints, vector<double>(nbOutputDims));

    int i = 0;
    for (auto const& k : selected)
    {
        for (int j = 0; j < nbOutputDims; ++j)
        {
            t[i][j] = possibleValues[k][j][i];
        }
        ++i;
    }

    return t;
}


template <typename ApproximatorType>
vector<vector<double>> ModelList<ApproximatorType>::predict(vector<vector<double>> const& points, vector<int> *modelIDs)
{
    return predict(points, true, modelIDs);
}


// Returns a string that details the forms in the list
template <typename ApproximatorType>
string ModelList<ApproximatorType>::toString(vector<string> inputNames, vector<string> outputNames) const
{
    auto const mEnd = models.end();
    auto mIt = models.begin();

    stringstream result;
    int k = 0;
    if (mIt != mEnd)
    {
        auto const forms = (*mIt)->getForms();
        result << "model " << k << ":" << endl;
        for (int i = 0; i < nbOutputDims; ++i)
        {
            result << outputNames[i] << ": " << approximator.formToString(forms[i], inputNames) << endl;
        }
        ++k; ++mIt;
    }

    for (; mIt != mEnd; ++k, ++mIt)
    {
        auto const forms = (*mIt)->getForms();
        result << "model " << k << ":" << endl;
        for (int i = 0; i < nbOutputDims; ++i)
        {
            result << outputNames[i] << ": " << approximator.formToString(forms[i], inputNames) << endl;
        }
    }

    return result.str();
}



}
