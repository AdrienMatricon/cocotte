
#include <algorithm>
#include <list>
#include <deque>
#include <queue>
#include <vector>
#include <set>
#include <unordered_set>
#include <utility>
#include <memory>
#include <string>
#include <sstream>
#include <iostream>
#include <cocotte/approximators/approximator.h>
#include <cocotte/models/models.hh>
#include <cocotte/modellist.h>

namespace Cocotte {



template <typename ApproximatorType>
ApproximatorType ModelList<ApproximatorType>::approximator;



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
void ModelList<ApproximatorType>::addModel(std::shared_ptr<Models::Model> model)
{
    models.push_back(model);
    ++nbModels;
    trainedClassifier = false;
}


template <typename ApproximatorType>
std::shared_ptr<Models::Model> ModelList<ApproximatorType>::firstModel()
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
// when adding points one by one, in batches, or all at once,
// except if noRollback (previous merges are kept)
// or addToExistingModelsOnly (new leaves merged into old models first) is set to true.
// In that case, merging goes faster and new leaves/nodes are marked as temporary
template <typename ApproximatorType>
void ModelList<ApproximatorType>::addPoint(std::shared_ptr<DataPoint const> pointAddress, bool noRollback)
{
    using std::move;
    using std::list;
    using std::shared_ptr;

    models = move(mergeAsMuchAsPossible(list<shared_ptr<Models::Model>>(1, createLeaf(pointAddress, noRollback)),
                                        move(models),
                                        noRollback,
                                        true));

    trainedClassifier = false;
    nbModels = models.size();
}


template <typename ApproximatorType>
void ModelList<ApproximatorType>::addPoints(std::vector<std::shared_ptr<DataPoint const>> const& pointAddresses,
                                            bool noRollback,
                                            bool addToExistingModelsOnly)
{
    using std::move;
    using std::list;
    using std::shared_ptr;

    bool const markAsTemporary = (noRollback || addToExistingModelsOnly);

    list<shared_ptr<Models::Model>> newLeaves;
    for (auto const& pointAddress : pointAddresses)
    {
        newLeaves.push_back(createLeaf(pointAddress, markAsTemporary));
    }

    models = move(mergeAsMuchAsPossible(move(newLeaves), move(models), noRollback, addToExistingModelsOnly));

    trainedClassifier = false;
    nbModels = models.size();
}



// Removes temporary models and add all points in temporary leaves with addPoints()
// (with noRollback and addToExistingModelsOnly set to false)
template <typename ApproximatorType>
void ModelList<ApproximatorType>::restructureModels()
{
    using std::vector;
    using std::list;
    using std::shared_ptr;
    using std::static_pointer_cast;
    using Models::Model;
    using Models::Leaf;
    using Models::Node;


    trainedClassifier = false;

    list<shared_ptr<Model>> toProcess = move(models);
    models = list<shared_ptr<Model>>{};
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
    using std::vector;
    using cv::Mat;
    using cv::RandomTreeParams;

    int const nbVars = Models::pointsBegin(models.front())->x.size();

    vector<float> priors(nbModels, 1.f);
    RandomTreeParams params(nbModels*2,                         // max depth
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


    Mat data(nbPoints, nbVars, CV_32F);
    Mat classification(nbPoints,1, CV_32F);

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

    classifier.train(data, CV_ROW_SAMPLE, classification, Mat(), Mat(), Mat(), Mat(), params);
    trainedClassifier = true;
}


template <typename ApproximatorType>
std::vector<int> ModelList<ApproximatorType>::selectModels(std::vector<std::vector<double>> const& points)
{
    using cv::Mat;
    using std::vector;

    size_t const nbPoints = points.size();
    size_t const nbVars = points[0].size();
    vector<int> result(nbPoints);

    if (!trainedClassifier)
    {
        trainClassifier();
    }

    for (size_t i = 0; i < nbPoints; ++i)
    {
        Mat point(1, nbVars, CV_32F);
        for (size_t j = 0; j < nbVars; ++j)
        {
            point.at<float>(0,j) = points[i][j];
        }

        result[i] = static_cast<int>(classifier.predict(point) + 0.5f);
    }

    return result;
}


template <typename ApproximatorType>
std::vector<std::vector<double>> ModelList<ApproximatorType>::predict(std::vector<std::vector<double>> const& points,
                                                                      bool fillModelIDs,
                                                                      std::vector<int> *modelIDs)
{
    using std::vector;

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
std::vector<std::vector<double>> ModelList<ApproximatorType>::predict(std::vector<std::vector<double>> const& points, std::vector<int> *modelIDs)
{
    return predict(points, true, modelIDs);
}


// Returns a string that details the forms in the list
template <typename ApproximatorType>
std::string ModelList<ApproximatorType>::toString(std::vector<std::string> inputNames, std::vector<std::string> outputNames) const
{
    using std::stringstream;

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



// Utility function
template <typename ApproximatorType>
std::shared_ptr<Models::Model> ModelList<ApproximatorType>::createLeaf(std::shared_ptr<DataPoint const> point, bool markAsTemporary)
{
    using std::vector;
    using std::shared_ptr;
    using Approximators::Form;
    using Models::Model;
    using Models::Leaf;

    vector<Form> forms;
    forms.reserve(nbOutputDims);

    for (auto const outDim : point->t[outputID])
    {
        forms.push_back(approximator.fitOnePoint(outDim.value, nbInputDims));
    }

    return shared_ptr<Model>(new Leaf(forms, point, markAsTemporary));
}


// Tries to merge two models into one without increasing complexity
// Returns the result if it succeeded, and a default-constructed shared_ptr otherwise
template <typename ApproximatorType>
std::shared_ptr<Models::Model> ModelList<ApproximatorType>::tryMerge(std::shared_ptr<Models::Model> model0, std::shared_ptr<Models::Model> model1, bool markAsTemporary)
{
    using std::min;
    using std::list;
    using std::shared_ptr;
    using std::static_pointer_cast;
    using std::const_pointer_cast;
    using Approximators::Form;
    using Models::Model;
    using Models::Leaf;
    using Models::Node;


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
            return shared_ptr<Model>{};
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
// - noRollback prevents those rollbacks if set to true
// - addToExistingModelsOnly prevents atomic models from being merged with each other
//   before being merged to models in independentlyMergedModels
template <typename ApproximatorType>
std::list<std::shared_ptr<Models::Model>> ModelList<ApproximatorType>::mergeAsMuchAsPossible(std::list<std::shared_ptr<Models::Model>>&& atomicModels,
                                                                                             std::list<std::shared_ptr<Models::Model>>&& independentlyMergedModels,
                                                                                             bool noRollback,
                                                                                             bool addToExistingModelsOnly)
{
    using std::list;
    using std::deque;
    using std::priority_queue;
    using std::set;
    using std::unordered_set;
    using std::pair;
    using std::shared_ptr;
    using std::static_pointer_cast;
    using Models::Model;
    using Models::Leaf;
    using Models::Node;
    using ModelPair = pair<shared_ptr<Model>,shared_ptr<Model>>;


    // Special cases
    if (atomicModels.empty())
    {
        // If there is no new model to merge,
        // all relevant merges were already done when the models were merged independently
        return independentlyMergedModels;
    }
    else if (independentlyMergedModels.empty()
             && (addToExistingModelsOnly
                 || (atomicModels.size() == 1)))
    {
        // There is nothing to do
        return atomicModels;
    }


    //
    // General case
    //


    // Some definitions first
    bool const markAsTemporary = (noRollback || addToExistingModelsOnly);

    set<shared_ptr<Model>> candidateModels;                                     // candidate models for the merging
    unordered_set<shared_ptr<Model>> unavailable;                               // models which have already been merged, or that have been rolled back

    deque<pair<double,shared_ptr<Model>>> independentMergesInnerDistances;      // nodes in independently merged models
    // and the distances between their children
    priority_queue<pair<double, ModelPair>,
            vector<pair<double, ModelPair>>,
            HasGreaterDistance<ModelPair>> candidateDistances;                  // candidate pairs of models, and the distances between them


    // We determine all candidate models for the merging phase,
    // and initialize candidateDistances as well as independentMergesInnerDistances
    {
        // First we consider independently merged models
        // and determine if we may need to roll back some merges
        if (noRollback)
        {
            candidateModels.insert(independentlyMergedModels.begin(), independentlyMergedModels.end());
            independentlyMergedModels.resize(0);
        }
        else
        {
            // We go down the tree as much as necessary
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

            // We sort queue by increasing distances
            sort(independentMergesInnerDistances.begin(), independentMergesInnerDistances.end(), pairCompareFirst<shared_ptr<Models::Model>>);
        }


        // Then we consider atomic models and compute distances
        if (addToExistingModelsOnly)
        {
            // We compute distances between the independently merged models and the atomic models
            for (auto cIt = candidateModels.begin(), cEnd = candidateModels.end(); cIt != cEnd; ++cIt)
            {
                for (auto aIt = atomicModels.begin(), aEnd = atomicModels.end(); aIt != aEnd; ++aIt)
                {
                    candidateDistances.push(make_pair(Models::getDistance(*cIt, *aIt, outputID),
                                                      make_pair(*cIt, *aIt)));
                }
            }

            // Then we move atomic models into the candidate models
            candidateModels.insert(atomicModels.begin(), atomicModels.end());
            atomicModels.resize(0);
        }
        else
        {
            // We moveadd atomic models to the candidate models
            candidateModels.insert(atomicModels.begin(), atomicModels.end());
            atomicModels.resize(0);

            // Then we compute all distances between candidate models
            for (auto cIt = candidateModels.begin(), cEnd = candidateModels.end(); cIt != cEnd; ++cIt)
            {

                auto otherIt = cIt; ++otherIt;
                for (; otherIt != cEnd; ++otherIt)
                {
                    candidateDistances.push(make_pair(Models::getDistance(*cIt, *otherIt, outputID),
                                                      make_pair(*cIt, *otherIt)));
                }
            }
        }
    }


    // Now that everything is initialized, we go on to do the merges
    double minDistCandidates = candidateDistances.top().first;

    double minDistIndependent = 0.;     // smallest distance between two models
    double maxDistIndependent = 0.;     // biggest distance between two models
    double supDistIndependent = 0.;     // even bigger distance (used to put entries at the end when sorting)

    if (!independentMergesInnerDistances.empty())
    {
        minDistIndependent = independentMergesInnerDistances.front().first;
        maxDistIndependent = independentMergesInnerDistances.back().first;
        supDistIndependent = maxDistIndependent*2 + 0.01;
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

            while (candidateDistances.empty() || (minDistIndependent <= minDistCandidates))
            {
                auto& entryDistance = iIt->first;
                auto const& mergedModel = iIt->second;
                auto const asNode = static_pointer_cast<Node>(mergedModel);
                auto const& child0 = asNode->getModel0();
                auto const& child1 = asNode->getModel1();

                if ((candidateModels.count(child0) > 0) && (candidateModels.count(child1) > 0))
                {
                    // The merge is validated
                    candidateModels.erase(child0);
                    candidateModels.erase(child1);
                    unavailable.insert(child0);
                    unavailable.insert(child1);

                    entryDistance = supDistIndependent;
                    ++nbPointsToTruncate;

                    // The model is added as a candidate model, and distances are added in candidateDistances
                    for (auto const& model : candidateModels)
                    {
                        candidateDistances.push(make_pair(Models::getDistance(mergedModel, model, outputID),
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

                        entryDistance = supDistIndependent;
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

                if (!independentMergesInnerDistances.empty())
                {
                    minDistIndependent = independentMergesInnerDistances.front().first;
                    maxDistIndependent = independentMergesInnerDistances.back().first;
                    supDistIndependent = maxDistIndependent*2 + 0.01;
                }
            }
        }


        // New Merges
        if (!candidateDistances.empty())
        {
            while (!candidateDistances.empty()
                   && (independentMergesInnerDistances.empty()
                       || (minDistCandidates < minDistIndependent)))
            {
                auto entry = candidateDistances.top();
                candidateDistances.pop();

                auto const& candidatePair = entry.second;
                auto const& model0 = candidatePair.first;
                auto const& model1 = candidatePair.second;

                if ((unavailable.count(model0) == 0) && (unavailable.count(model1) == 0))
                {
                    auto newModel = tryMerge(model0, model1, markAsTemporary);

                    if (newModel)
                    {
                        // Successful merge
                        candidateModels.erase(model0);
                        candidateModels.erase(model1);
                        unavailable.insert(model0);
                        unavailable.insert(model1);

                        // The model is added as a candidate model, and distances are added in candidateDistances
                        for (auto const& model : candidateModels)
                        {
                            candidateDistances.push(make_pair(Models::getDistance(newModel, model, outputID),
                                                              make_pair(newModel, model)));
                        }

                        candidateModels.insert(newModel);
                        break;
                    }
                }

                minDistCandidates = candidateDistances.top().first;
            }

            if (!candidateDistances.empty())
            {
                minDistCandidates = candidateDistances.top().first;
            }
        }
    }


    // Finally, we put everything in a list and return it
    return list<shared_ptr<Model>>(candidateModels.begin(), candidateModels.end());
}


}
