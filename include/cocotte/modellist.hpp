
#include <algorithm>
#include <list>
#include <deque>
#include <queue>
#include <vector>
#include <set>
#include <unordered_set>
#include <utility>
#include <tuple>
#include <memory>
#include <string>
#include <sstream>
#include <iostream>
#include <cocotte/approximators/approximator.h>
#include <cocotte/models/models.hh>
#include <cocotte/modellist.h>

namespace Cocotte {



// Constructor
template <typename ApproximatorType>
ModelList<ApproximatorType>::ModelList(unsigned int oID, unsigned int nId, unsigned int nOd): outputID(oID), nbInputDims(nId), nbOutputDims(nOd)
{}



// Accessors
template <typename ApproximatorType>
unsigned int ModelList<ApproximatorType>::getNbModels() const
{
    return nbModels;
}


template <typename ApproximatorType>
unsigned int ModelList<ApproximatorType>::getComplexity(unsigned int j) const
{
    unsigned int complexity = 0;

    for (auto const& model : models)
    {
        complexity += model->getForms()[j].complexity;
    }

    return complexity;
}



// Adding and removing models
template <typename ApproximatorType>
void ModelList<ApproximatorType>::addModel(std::shared_ptr<Models::Model<ApproximatorType>> model)
{
    models.push_back(model);
    ++nbModels;
    classifier.reset();
}


template <typename ApproximatorType>
std::shared_ptr<Models::Model<ApproximatorType>> ModelList<ApproximatorType>::firstModel()
{
    return models.front();
}


template <typename ApproximatorType>
void ModelList<ApproximatorType>::removeFirstModel()
{
    models.pop_front();
    --nbModels;
    classifier.reset();
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
    using ModelType = Models::Model<ApproximatorType>;

    models = move(mergeAsMuchAsPossible(
                      list<shared_ptr<ModelType>>(
                          1, createLeaf(pointAddress, noRollback)),
                      move(models),
                      noRollback,
                      true));

    classifier.reset();
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
    using ModelType = Models::Model<ApproximatorType>;

    bool const markAsTemporary = (noRollback || addToExistingModelsOnly);

    list<shared_ptr<ModelType>> newLeaves;
    for (auto const& pointAddress : pointAddresses)
    {
        newLeaves.push_back(createLeaf(pointAddress, markAsTemporary));
    }

    models = move(mergeAsMuchAsPossible(move(newLeaves), move(models), noRollback, addToExistingModelsOnly));

    classifier.reset();
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
    using ModelType = Models::Model<ApproximatorType>;
    using LeafType = Models::Leaf<ApproximatorType>;
    using NodeType = Models::Node<ApproximatorType>;


    classifier.reset();

    list<shared_ptr<ModelType>> toProcess = move(models);
    models = list<shared_ptr<ModelType>>{};
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
                pointsInTemporaryLeaves.push_back(static_pointer_cast<LeafType>(current)->getPointAddress());
            }
            else
            {
                shared_ptr<NodeType> asNode = static_pointer_cast<NodeType>(current);
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



// Checks if for each point there exists a model that could predict it
template <typename ApproximatorType>
bool ModelList<ApproximatorType>::canBePredicted(std::vector<std::shared_ptr<DataPoint const>> const& pointAddresses)
{
    using std::vector;

    for (auto const& pointAddress : pointAddresses)
    {
        vector<vector<double>> xVal;
        {
            vector<double> xValTemp;
            xValTemp.reserve(pointAddress->x.size());

            for (auto const& dim : pointAddress->x)
            {
                xValTemp.push_back(dim.value);
            }
            xVal.push_back(xValTemp);
        }

        vector<double> tVal, tPrec;
        {
            tVal.reserve(pointAddress->t[outputID].size());
            tPrec.reserve(tVal.size());

            for (auto const& dim : pointAddress->t[outputID])
            {
                tVal.push_back(dim.value);
                tPrec.push_back(dim.precision);
            }
        }

        bool cannotBePredicted = true;

        for (auto const& model : models)
        {
            bool predictedByThisModel = true;
            unsigned int i = 0;

            for (auto const& f : model->getForms())
            {
                auto prediction = ApproximatorType::estimate(f, xVal)[0];

                if (abs(prediction - tVal[i]) > tPrec[i])
                {
                    predictedByThisModel = false;
                    break;
                }

                ++i;
            }

            if (predictedByThisModel)
            {
                cannotBePredicted = false;
                break;
            }
        }

        if (cannotBePredicted)
        {
            return false;
        }
    }

    return true;
}



// Predict the value of a point
template <typename ApproximatorType>
void ModelList<ApproximatorType>::trainClassifier()
{
    using std::vector;
    using cv::Mat;
    using cv::RandomTreeParams;

    unsigned int const nbVars = Models::pointsBegin<ApproximatorType>(models.front())->x.size();

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

    unsigned int nbPoints = 0;
    for (auto const& model : models)
    {
        nbPoints += model->getNbPoints();
    }


    Mat data(nbPoints, nbVars, CV_32F);
    Mat classification(nbPoints,1, CV_32F);

    {
        float label = 0.f;
        unsigned int i = 0;
        for (auto const& model : models)
        {
            auto const mEnd = Models::pointsEnd<ApproximatorType>(model);
            for (auto mIt = Models::pointsBegin<ApproximatorType>(model); mIt != mEnd; ++mIt, ++i)
            {
                auto const& x = mIt->x;
                for (unsigned int j = 0; j < nbVars; ++j)
                {
                    data.at<float>(i,j) = x[j].value;
                }

                classification.at<float>(i, 0) = label;
            }

            label += 1.0f;
        }
    }

    classifier = std::shared_ptr<cv::RandomTrees>(new cv::RandomTrees);
    classifier->train(data, CV_ROW_SAMPLE, classification, Mat(), Mat(), Mat(), Mat(), params);
}


template <typename ApproximatorType>
std::vector<unsigned int> ModelList<ApproximatorType>::selectModels(std::vector<std::vector<double>> const& points)
{
    using cv::Mat;
    using std::vector;

    unsigned int const nbPoints = points.size();
    unsigned int const nbVars = points[0].size();
    vector<unsigned int> result(nbPoints);

    if (!classifier)
    {
        trainClassifier();
    }

    for (unsigned int i = 0; i < nbPoints; ++i)
    {
        Mat point(1, nbVars, CV_32F);
        for (unsigned int j = 0; j < nbVars; ++j)
        {
            point.at<float>(0,j) = points[i][j];
        }

        result[i] = static_cast<unsigned int>(classifier->predict(point) + 0.5f);
    }

    return result;
}


template <typename ApproximatorType>
std::vector<std::vector<double>> ModelList<ApproximatorType>::predict(std::vector<std::vector<double>> const& points,
                                                                      bool fillModelIDs,
                                                                      std::vector<unsigned int> *modelIDs)
{
    using std::vector;

    unsigned int const nbPoints = points.size();
    vector<vector<vector<double>>> possibleValues;
    possibleValues.reserve(nbModels);

    for (auto const& model : models)
    {
        auto const forms = model->getForms();
        vector<vector<double>> estimates;
        estimates.reserve(nbOutputDims);
        for (auto const& f : forms)
        {
            estimates.push_back(ApproximatorType::estimate(f, points));
        }
        possibleValues.push_back(estimates);
    }

    vector<unsigned int> const selected = selectModels(points);
    if (fillModelIDs)
    {
        *modelIDs = selected;
    }

    vector<vector<double>> t(nbPoints, vector<double>(nbOutputDims));

    unsigned int i = 0;
    for (auto const& k : selected)
    {
        for (unsigned int j = 0; j < nbOutputDims; ++j)
        {
            t[i][j] = possibleValues[k][j][i];
        }
        ++i;
    }

    return t;
}


template <typename ApproximatorType>
std::vector<std::vector<double>> ModelList<ApproximatorType>::predict(std::vector<std::vector<double>> const& points, std::vector<unsigned int> *modelIDs)
{
    return predict(points, true, modelIDs);
}


// Returns a string that details the forms in the list
template <typename ApproximatorType>
std::string ModelList<ApproximatorType>::toString(std::vector<std::string> inputNames, std::vector<std::string> outputNames) const
{
    using std::stringstream;
    using std::endl;

    auto const mEnd = models.end();
    auto mIt = models.begin();

    stringstream result;
    unsigned int k = 0;
    if (mIt != mEnd)
    {
        auto const forms = (*mIt)->getForms();
        result << "model " << k << ":" << endl;
        for (unsigned int i = 0; i < nbOutputDims; ++i)
        {
            result << outputNames[i] << ": " << ApproximatorType::formToString(forms[i], inputNames) << endl;
        }
        ++k; ++mIt;
    }

    for (; mIt != mEnd; ++k, ++mIt)
    {
        auto const forms = (*mIt)->getForms();
        result << "model " << k << ":" << endl;
        for (unsigned int i = 0; i < nbOutputDims; ++i)
        {
            result << outputNames[i] << ": " << ApproximatorType::formToString(forms[i], inputNames) << endl;
        }
    }

    return result.str();
}



// Utility function
template <typename ApproximatorType>
std::shared_ptr<Models::Model<ApproximatorType>> ModelList<ApproximatorType>::createLeaf(
        std::shared_ptr<DataPoint const> point, bool markAsTemporary)
{
    using std::vector;
    using std::shared_ptr;
    using Approximators::Form;
    using ModelType = Models::Model<ApproximatorType>;
    using LeafType = Models::Leaf<ApproximatorType>;
    using FormType = Approximators::Form<ApproximatorType>;

    vector<FormType> forms;
    forms.reserve(nbOutputDims);

    for (auto const outDim : point->t[outputID])
    {
        forms.push_back(ApproximatorType::fitOnePoint(outDim.value, nbInputDims));
    }

    return shared_ptr<ModelType>(new LeafType(forms, point, markAsTemporary));
}


// Tries to merge two models into one without increasing complexity
// Returns the result if it succeeded, and a default-constructed shared_ptr otherwise
template <typename ApproximatorType>
std::shared_ptr<Models::Model<ApproximatorType>> ModelList<ApproximatorType>::tryMerge(
        std::shared_ptr<Models::Model<ApproximatorType>> model0,
        std::shared_ptr<Models::Model<ApproximatorType>> model1,
        bool markAsTemporary)
{
    using std::max;
    using std::vector;
    using std::list;
    using std::shared_ptr;
    using std::static_pointer_cast;
    using std::const_pointer_cast;
    using Approximators::Form;
    using ModelType = Models::Model<ApproximatorType>;
    using NodeType = Models::Node<ApproximatorType>;
    using FormType = Approximators::Form<ApproximatorType>;

    // We determine if new node should be temporary
    markAsTemporary = (markAsTemporary || model0->isTemporary() || model1->isTemporary());

    shared_ptr<ModelType> node (new NodeType(model0, model1, markAsTemporary));
    unsigned int const nbPoints = node->getNbPoints();
    auto const mBegin = Models::pointsBegin<ApproximatorType>(const_pointer_cast<ModelType const>(node));
    auto const mEnd = Models::pointsEnd<ApproximatorType>(const_pointer_cast<ModelType const>(node));

    vector<FormType> const forms0 = model0->getForms(), forms1 = model1->getForms();
    vector<FormType> newForms;

    for (unsigned int outputDim = 0; outputDim < nbOutputDims; ++outputDim)
    {
        FormType const form0 = forms0[outputDim], form1 = forms1[outputDim];
        unsigned int const totalNbDimensions = form0.usedDimensions.getTotalNbDimensions();

//        UsedDimensions availableDimensions = UsedDimensions::allDimensions(totalNbDimensions);
        UsedDimensions availableDimensions = form0.relevantDimensions + form1.relevantDimensions;
        unsigned int const nbUnusedDimensions = availableDimensions.getNbUnused();
        unsigned int const maxAllowedComplexity = form0.complexity + form1.complexity;

        // The lowest complexity form which has been found
        list<FormType> bestForms;
        unsigned int bestComplexity = maxAllowedComplexity + 2;

        // We look for a form that would fit the points,
        //   and use a binary search to get the lowest complexity one
        // We don't immediately test the highest complexity
        //   because the computational cost can be prohibitive
        for (unsigned int nbAdditionalDimensions = 0; nbAdditionalDimensions <= nbUnusedDimensions; ++nbAdditionalDimensions)
        {
            unsigned int lowerBound = (nbAdditionalDimensions == 0) ?
                        max(form0.complexity, form1.complexity) : 1;
            unsigned int upperBound = maxAllowedComplexity;

            // Instead of trying one complexity in [lowerBound, upperBound], we will consider a range,
            //   so that we don't miss forms with lower complexities but different dimensions.
            // That way, if no form in the range fits, we know the complexity was not high enough
            unsigned int middleComplexityRangeMin,  middleComplexityRangeMax;

            while (upperBound > lowerBound)
            {
                if (upperBound - lowerBound < 5)
                {
                    // We finish in one go
                    middleComplexityRangeMax = upperBound;
                    middleComplexityRangeMin = lowerBound;
                }
                else
                {
                    middleComplexityRangeMax = (lowerBound + upperBound) / 2;

                    if (lowerBound == 1)
                    {
                        middleComplexityRangeMin = 1;
                    }
                    else
                    {
                        middleComplexityRangeMin =
                                ApproximatorType::getComplexityRangeLowerBound(totalNbDimensions, middleComplexityRangeMax);
                    }
                }

                // We look for possible forms with complexity in [middleComplexityRangeMin, middleComplexityRangeMax]
                //   which use exactly nbAdditionalDimensions dimensions not in availableDimensions
                auto possibleForms = ApproximatorType::getFormsInComplexityRange(availableDimensions,
                                                                                 nbAdditionalDimensions,
                                                                                 middleComplexityRangeMin,
                                                                                 middleComplexityRangeMax);

                bool success = false;

                // We try to fit the points with those forms,
                //   starting with the lowest complexity forms
                {
                    for (auto& someForms : possibleForms)
                    {
                        if (someForms.front().complexity > bestComplexity)
                        {
                            break;
                        }

                        for (auto& form : someForms)
                        {
                            // For each form, we try to fit all points
                            bool const fitResult = ApproximatorType::tryFit(form, nbPoints, mBegin, mEnd, outputID, outputDim);
                            if (fitResult)
                            {
                                if (!success)
                                {
                                    success = true;
                                    upperBound = middleComplexityRangeMin - 1u;
                                }

                                if (form.complexity < bestComplexity)
                                {
                                    bestForms = list<FormType>{};
                                    bestComplexity = form.complexity;
                                }

                                bestForms.push_back(form);
                            }
                        }
                    }
                }

                if (!success)
                {
                    lowerBound = middleComplexityRangeMax + 1u;
                }
            }

            if (bestComplexity < maxAllowedComplexity)
            {
                // At least one success and structure was used to reduce complexity,
                //   so we can end here.
                // Otherwise we go for the next loop, with more additional dimensions
                break;
            }
        }

        if (bestComplexity > maxAllowedComplexity)
        {
            // We never succeeded
            // => we return a default-constructed shared_ptr
            return shared_ptr<ModelType>{};
        }

        auto bestNewForm = bestForms.front();
        bestForms.pop_front();
        auto bestFitness = ApproximatorType::refine(bestNewForm, nbPoints, mBegin, mEnd, outputID, outputDim);

        for (auto& form : bestForms)
        {
            auto const fitness = ApproximatorType::refine(form, nbPoints, mBegin, mEnd, outputID, outputDim);

            if (fitness >= bestFitness)
            {
                if (fitness > bestFitness)
                {
                    bestNewForm = form;
                    bestFitness = fitness;
                }
                else
                {
                    bestNewForm.relevantDimensions += form.usedDimensions;
                }
            }
        }

        newForms.push_back(bestNewForm);
    }

    static_pointer_cast<NodeType>(node)->setForms(newForms);
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
std::list<std::shared_ptr<Models::Model<ApproximatorType>>> ModelList<ApproximatorType>::mergeAsMuchAsPossible(
        std::list<std::shared_ptr<Models::Model<ApproximatorType>>>&& atomicModels,
        std::list<std::shared_ptr<Models::Model<ApproximatorType>>>&& independentlyMergedModels,
        bool noRollback,
        bool addToExistingModelsOnly)
{
    using std::vector;
    using std::list;
    using std::deque;
    using std::priority_queue;
    using std::set;
    using std::unordered_set;
    using std::pair;
    using std::shared_ptr;
    using std::static_pointer_cast;
    using ModelType = Models::Model<ApproximatorType>;
    using NodeType = Models::Node<ApproximatorType>;
    using ModelPair = pair<shared_ptr<ModelType>,shared_ptr<ModelType>>;

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

    set<shared_ptr<ModelType>> candidateModels;                                     // candidate models for the merging
    unordered_set<shared_ptr<ModelType>> unavailable;                               // models which have already been merged, or that have been rolled back

    deque<pair<Models::ModelDistance,shared_ptr<ModelType>>> independentMergesInnerDistances;      // nodes in independently merged models
    // and the distances between their children
    priority_queue<pair<Models::ModelDistance, ModelPair>,
            vector<pair<Models::ModelDistance, ModelPair>>,
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
                    auto const asNode = static_pointer_cast<NodeType>(model);
                    auto const biggestInnerDistance = asNode->getBiggestInnerDistance(outputID);

                    for (auto const& atomic : atomicModels)
                    {
                        // If the atomic models are closer to the model than some merged models were to each other,
                        // we may need to roll back some merges
                        if (Models::getDistance<ApproximatorType>(model, atomic, outputID) < biggestInnerDistance)
                        {
                            auto const& child0 = asNode->getModel0();
                            auto const& child1 = asNode->getModel1();

                            mayBeUnmerged = true;
                            auto const innerDistance = Models::getDistance<ApproximatorType>(child0, child1, outputID);
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
            sort(independentMergesInnerDistances.begin(), independentMergesInnerDistances.end(),
                 pairCompareFirst<Models::ModelDistance, shared_ptr<ModelType>>);
        }


        // Then we consider atomic models and compute distances
        if (addToExistingModelsOnly)
        {
            // We compute distances between the independently merged models and the atomic models
            for (auto cIt = candidateModels.begin(), cEnd = candidateModels.end(); cIt != cEnd; ++cIt)
            {
                for (auto aIt = atomicModels.begin(), aEnd = atomicModels.end(); aIt != aEnd; ++aIt)
                {
                    candidateDistances.push(make_pair(Models::getDistance<ApproximatorType>(*cIt, *aIt, outputID),
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
                    candidateDistances.push(make_pair(Models::getDistance<ApproximatorType>(*cIt, *otherIt, outputID),
                                                      make_pair(*cIt, *otherIt)));
                }
            }
        }
    }


    // Now that everything is initialized, we go on to do the merges
    auto minDistCandidates = candidateDistances.top().first;

    Models::ModelDistance minDistIndependent;   // smallest distance between two models
    Models::ModelDistance supDistIndependent;   // big distance used to put entries at the end when sorting

    if (!independentMergesInnerDistances.empty())
    {
        minDistIndependent = independentMergesInnerDistances.front().first;
        supDistIndependent = Models::ModelDistance::getBiggerDistanceThan(independentMergesInnerDistances.back().first);
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
            unsigned int nbPointsToTruncate = 0;

            bool doneSomething = false;

            while (candidateDistances.empty() || !(minDistCandidates < minDistIndependent))
            {
                auto& entryDistance = iIt->first;
                auto const& mergedModel = iIt->second;
                auto const asNode = static_pointer_cast<NodeType>(mergedModel);
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
                        candidateDistances.push(make_pair(Models::getDistance<ApproximatorType>(mergedModel, model, outputID),
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
                sort(independentMergesInnerDistances.begin(), independentMergesInnerDistances.end(),
                     pairCompareFirst<Models::ModelDistance, shared_ptr<ModelType>>);

                if (nbPointsToTruncate > 0)
                {
                    independentMergesInnerDistances.resize(independentMergesInnerDistances.size() - nbPointsToTruncate);
                }

                if (!independentMergesInnerDistances.empty())
                {
                    minDistIndependent = independentMergesInnerDistances.front().first;
                    supDistIndependent = Models::ModelDistance::getBiggerDistanceThan(independentMergesInnerDistances.back().first);
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
                            candidateDistances.push(make_pair(Models::getDistance<ApproximatorType>(newModel, model, outputID),
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
    return list<shared_ptr<ModelType>>(candidateModels.begin(), candidateModels.end());
}


}
