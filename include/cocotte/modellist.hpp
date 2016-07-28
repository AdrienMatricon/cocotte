
#include <algorithm>
using std::min;
using std::max;
#include <list>
using std::list;
#include <vector>
using std::vector;
#include <set>
using std::set;
#include <boost/unordered_set.hpp>
using boost::unordered_set;
#include <utility>
using std::pair;
using std::make_pair;
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



// Creates leaves for the new points and merges them with models or submodels,
// starting with the closest ones. We expect to get the same result
// when adding points one by one, in batches, or all at once.
template <typename ApproximatorType>
void ModelList<ApproximatorType>::addPoint(boost::shared_ptr<DataPoint const> pointAddress)
{
    addPoints(vector<boost::shared_ptr<DataPoint const>>(1, pointAddress));
}


template <typename ApproximatorType>
void ModelList<ApproximatorType>::addPoints(vector<boost::shared_ptr<DataPoint const>> const& pointAddresses)
{
    trainedClassifier = false;

    list<shared_ptr<Model>> newLeaves;

    for (auto const& pointAddress : pointAddresses)
    {
        newLeaves.push_back(createLeaf(pointAddress));
    }

    models = std::move(mergeAsMuchAsPossible(std::move(newLeaves), std::move(models)));
    nbModels = models.size();
}



// Creates a leaf for a new point and tries to merge it with an already existing model,
// Models created in this way are marked as temporary
// Returns true if it succeeded, false if a new model was created
template <typename ApproximatorType>
bool ModelList<ApproximatorType>::tryAddingPointToExistingModels(boost::shared_ptr<DataPoint const> pointAddress)
{
    size_t const oldNbModels = nbModels;

    trainedClassifier = false;

    models.push_back(createLeaf(pointAddress, true));
    models = std::move(mergeAsMuchAsPossible(std::move(models), list<shared_ptr<Model>>(), true));
    nbModels = models.size();

    return (nbModels == oldNbModels);
}



// Removes temporary models and add all points in temporary leaves with addPoints()
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

        // End of the scope, current is destroyed.
        // It was the last shared_ptr to this model, which is therefore destroyed.
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
vector<vector<double>> ModelList<ApproximatorType>::predict(vector<vector<double>> const& points, bool fillModelIDs, vector<int> *modelIDs)
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
    if (fillModelIDs)
    {
        *modelIDs = selected;
    }

    vector<vector<double>> t(nbPoints, vector<double>(nbOutputDims));

    size_t i = 0;
    for (auto const& k : selected)
    {
        for (size_t j = 0; j < nbOutputDims; ++j)
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
        for (size_t i = 0; i < nbOutputDims; ++i)
        {
            result << outputNames[i] << ": " << approximator.formToString(forms[i], inputNames) << endl;
        }
        ++k; ++mIt;
    }

    for (; mIt != mEnd; ++k, ++mIt)
    {
        auto const forms = (*mIt)->getForms();
        result << "model " << k << ":" << endl;
        for (size_t i = 0; i < nbOutputDims; ++i)
        {
            result << outputNames[i] << ": " << approximator.formToString(forms[i], inputNames) << endl;
        }
    }

    return result.str();
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

    for (size_t dim = 0; dim < nbOutputDims; ++dim)
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


// Merges the models with each other, starting with the closest ones:
// - atomicModels is a list of leaves, or more generally of models that are supposed correctly merged
// - independentlyMergedModels is a list of models resulting from previous merges
//   => Those merges may be rolled back because of the new models in atomicModels.
//      We expect to get the same result with independentlyMergedModels or with the
//      list of every leaf in independentlyMergedModels
template <typename ApproximatorType>
list<shared_ptr<Model>> ModelList<ApproximatorType>::mergeAsMuchAsPossible(list<shared_ptr<Model>>&& atomicModels,
                                                                           list<shared_ptr<Model>>&& independentlyMergedModels,
                                                                           bool markMergesAsTemporary)
{
    using ModelPair = pair<shared_ptr<Model>,shared_ptr<Model>>;

    // Special cases
    if (atomicModels.empty())
    {
        // If there is no new model to merge,
        // all relevant merges were already done when the models were merged independently
        return independentlyMergedModels;
    }
    else if(independentlyMergedModels.empty() && (atomicModels.size() == 1))
    {
        // If there is only one model and it is atomic, there is nothing to do
        return atomicModels;
    }

    // We declare what we will need for handling all models
    vector<pair<double,shared_ptr<Model>>> independentMergesInnerDistances;     // nodes in independently merged models
    // and the distances between their children

    set<shared_ptr<Model>> candidateModels;                                     // candidate models for the merging
    unordered_set<shared_ptr<Model>> unavailable;                                    // models which have already been merged, or that have been rolled back
    vector<pair<double, ModelPair>> candidateDistances;                         // candidate pairs of models, and the distances between them


    // We determine all candidate models for the merging phase,
    // and initialize candidateDistances as well as independentMergesInnerDistances
    {
        // First we consider distances between independently merged models (or their submodels)
        // and atomic models, to see if we may need to roll back some merges
        {
            while (!independentlyMergedModels.empty())
            {
                auto const model = independentlyMergedModels.back();
                independentlyMergedModels.pop_back();

                bool mayBeUnmerged = false;

                if (!model->isLeaf())
                {
                    auto const asNode = static_pointer_cast<Node>(model);
                    auto const biggestInnerDistance = asNode->getBiggestInnerDistance(outputID);

                    for (auto const& atomic : atomicModels)
                    {
                        // If the atomic models are closer to the model than some merged models were to each other,
                        // we may need to roll back some merges
                        if (Models::getDistance(model, atomic, outputID) < biggestInnerDistance)
                        {
                            auto const& child0 = asNode->getModel0();
                            auto const& child1 = asNode->getModel1();

                            mayBeUnmerged = true;
                            double const innerDistance = Models::getDistance(child0, child1, outputID);
                            independentMergesInnerDistances.push_back(make_pair(innerDistance, model));

                            independentlyMergedModels.push_back(child0);
                            independentlyMergedModels.push_back(child1);

                            break;
                        }
                    }
                }

                // Otherwise (or if we reached a leaf), we add the model to the set of candidate models
                if (!mayBeUnmerged)
                {
                    candidateModels.insert(model);
                }

            }
        }


        // Then we add atomic models to the candidate models
        candidateModels.insert(atomicModels.begin(), atomicModels.end());
        atomicModels.resize(0);


        // Finally we compute all distances between candidate models
        for (auto cIt = candidateModels.begin(), cEnd = candidateModels.end(); cIt != cEnd; ++cIt)
        {

            auto otherIt = cIt; ++otherIt;
            for (; otherIt != cEnd; ++otherIt)
            {
                candidateDistances.push_back(make_pair(Models::getDistance(*cIt, *otherIt, outputID),
                                                       make_pair(*cIt, *otherIt)));
            }
        }



        // Now we sort queues by increasing distances
        sort(candidateDistances.begin(), candidateDistances.end(), pairCompareFirst<ModelPair>);
        sort(independentMergesInnerDistances.begin(), independentMergesInnerDistances.end(), pairCompareFirst<shared_ptr<Model>>);
    }


    // Now that everything is initialized, we go on to do the merges
    double minDistCandidates = candidateDistances.front().first;
    double maxDistCandidates = candidateDistances.back().first;

    double maxDist = maxDistCandidates;     // biggest distance between two models
    double supDist = maxDist*2 + 0.01;      // even bigger distance (used to put entries at the end when sorting)

    double minDistIndependent = supDist;
    double maxDistIndependent = supDist;

    if (!independentMergesInnerDistances.empty())
    {
        minDistIndependent = independentMergesInnerDistances.front().first;
        maxDistIndependent = independentMergesInnerDistances.back().first;

        maxDist = max(maxDistCandidates, maxDistIndependent);
        supDist = maxDist*2 + 0.01;
    }

    while (!independentMergesInnerDistances.empty() || !candidateDistances.empty())
    {
        // Merges that are not rolled back
        if (!independentMergesInnerDistances.empty())
        {
            // We update this because we may have increased minDistIndependent
            // during the previous iteration if the children nodes of the
            // first entries were not candidate models
            minDistIndependent = independentMergesInnerDistances.front().first;

            auto iIt = independentMergesInnerDistances.begin();
            auto const iEnd = independentMergesInnerDistances.end();
            size_t nbPointsToTruncate = 0;

            bool doneSomething = false;

            while (minDistIndependent <= minDistCandidates)
            {
                auto& entryDistance = iIt->first;
                auto const& mergedModel = iIt->second;
                auto const asNode = static_pointer_cast<Node>(mergedModel);
                auto const& child0 = asNode->getModel0();
                auto const& child1 = asNode->getModel1();

                if ((candidateModels.count(child0) > 0) && (candidateModels.count(child1) > 0))
                {
                    // The merge is validated
                    candidateModels.erase(candidateModels.find(child0));
                    candidateModels.erase(candidateModels.find(child1));
                    unavailable.insert(child0);
                    unavailable.insert(child1);

                    entryDistance = supDist;
                    ++nbPointsToTruncate;

                    // The model is added as a candidate model, and distances are added in candidateDistances
                    for (auto const& model : candidateModels)
                    {
                        candidateDistances.push_back(make_pair(Models::getDistance(mergedModel, model, outputID),
                                                               make_pair(mergedModel, model)));
                    }

                    candidateModels.insert(mergedModel);
                    doneSomething = true;
                    break;
                }
                else
                {
                    if ((unavailable.count(child0) > 0) || (unavailable.count(child1) > 0))
                    {
                        // The merge is definitively rolled back (otherwise it will be decided later)
                        unavailable.insert(mergedModel);

                        entryDistance = supDist;
                        ++nbPointsToTruncate;
                        doneSomething = true;
                    }

                    ++iIt;

                    if (iIt == iEnd)
                    {
                        // End of the queue
                        break;
                    }

                    minDistIndependent = iIt->first;
                }
            }

            if (doneSomething)
            {
                sort(independentMergesInnerDistances.begin(), independentMergesInnerDistances.end(), pairCompareFirst<shared_ptr<Model>>);

                if (nbPointsToTruncate > 0)
                {
                    independentMergesInnerDistances.resize(independentMergesInnerDistances.size() - nbPointsToTruncate);
                }

                if (independentMergesInnerDistances.empty())
                {
                    minDistIndependent = supDist;
                }
                else
                {
                    minDistIndependent = independentMergesInnerDistances.front().first;
                    maxDistIndependent = independentMergesInnerDistances.back().first;
                    maxDist = max(maxDistCandidates, maxDistIndependent);
                    supDist = maxDist*2 + 0.01;
                }
            }
        }


        // New Merges
        if (!candidateDistances.empty())
        {
            auto cIt = candidateDistances.begin();
            auto const cEnd = candidateDistances.end();
            size_t nbPointsToTruncate = 0;

            while (minDistCandidates < minDistIndependent)
            {
                auto& entryDistance = cIt->first;
                auto const& candidatePair = cIt->second;
                auto const& model0 = candidatePair.first;
                auto const& model1 = candidatePair.second;

                entryDistance = supDist;
                ++nbPointsToTruncate;

                if ((unavailable.count(model0) == 0) && (unavailable.count(model1) == 0))
                {
                    auto newModel = tryMerge(model0, model1, markMergesAsTemporary);

                    if (newModel)
                    {
                        // Successful merge
                        candidateModels.erase(candidateModels.find(model0));
                        candidateModels.erase(candidateModels.find(model1));
                        unavailable.insert(model0);
                        unavailable.insert(model1);

                        // The model is added as a candidate model, and distances are added in candidateDistances
                        for (auto const& model : candidateModels)
                        {
                            candidateDistances.push_back(make_pair(Models::getDistance(newModel, model, outputID),
                                                                   make_pair(newModel, model)));
                        }

                        candidateModels.insert(newModel);
                        break;
                    }
                }

                ++cIt;

                if (cIt == cEnd)
                {
                    // End of the queue
                    break;
                }

                minDistCandidates = cIt->first;
            }

            sort(candidateDistances.begin(), candidateDistances.end(), pairCompareFirst<ModelPair>);

            if (nbPointsToTruncate > 0)
            {
                candidateDistances.resize(candidateDistances.size() - nbPointsToTruncate);
            }

            if (candidateDistances.empty())
            {
                minDistCandidates = supDist;
            }
            else
            {
                minDistCandidates = candidateDistances.front().first;
                maxDistCandidates = candidateDistances.back().first;
                maxDist = max(maxDistCandidates, maxDistIndependent);
                supDist = maxDist*2 + 0.01;
            }
        }
    }


    // Finally, we put everything in a list and return it
    return list<shared_ptr<Model>>(candidateModels.begin(), candidateModels.end());
}


}
