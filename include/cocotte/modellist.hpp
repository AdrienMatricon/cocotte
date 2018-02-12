
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
ModelList<ApproximatorType>::ModelList(unsigned int oID, unsigned int nId, unsigned int nOd):
    outputID(oID), nbInputDims(nId), nbOutputDims(nOd)
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
void ModelList<ApproximatorType>::removeModel(std::shared_ptr<Models::Model<ApproximatorType>> model)
{
    auto const mEnd = models.end();
    for (auto mIt = models.begin(); mIt != mEnd; ++mIt)
    {
        if (*mIt == model)
        {
            models.erase(mIt);
            break;
        }
    }

    --nbModels;
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
// starting with the closest ones
template <typename ApproximatorType>
void ModelList<ApproximatorType>::addPoint(std::shared_ptr<DataPoint const> pointAddress)
{
    addModel(createLeaf(pointAddress));
    performPointStealing();
}


template <typename ApproximatorType>
void ModelList<ApproximatorType>::addPoints(std::vector<std::shared_ptr<DataPoint const>> const& pointAddresses)
{
    for (auto const& pointAddress : pointAddresses)
    {
        addModel(createLeaf(pointAddress));
    }

    performPointStealing();
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
std::vector<std::vector<double>> ModelList<ApproximatorType>::predict(
        std::vector<std::vector<double>> const& points,
        std::vector<unsigned int> *modelIDs)
{
    return predict(points, true, modelIDs);
}


// Returns a string that details the forms in the list
template <typename ApproximatorType>
std::string ModelList<ApproximatorType>::toString(
        std::vector<std::string> inputNames,
        std::vector<std::string> outputNames) const
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
        std::shared_ptr<DataPoint const> point)
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

    return shared_ptr<ModelType>(new LeafType(forms, point));
}


// - Tries to merge models into one without increasing complexity
// - Returns the result if it succeeded, and a default-constructed shared_ptr otherwise
// - shouldWork can be set to specify complexities and dimensions that should allow fitting,
//    to make sure that a solution will be found
template <typename ApproximatorType>
std::shared_ptr<Models::Model<ApproximatorType>> ModelList<ApproximatorType>::tryMerge(
        std::vector<std::shared_ptr<Models::Model<ApproximatorType>>> candidateModels,
        std::vector<std::pair<unsigned int, UsedDimensions>> const& shouldWork)
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

    shared_ptr<ModelType> node (new NodeType(candidateModels));
    unsigned int const nbPoints = node->getNbPoints();
    auto const mBegin = Models::pointsBegin<ApproximatorType>(const_pointer_cast<ModelType const>(node));
    auto const mEnd = Models::pointsEnd<ApproximatorType>(const_pointer_cast<ModelType const>(node));

    vector<FormType> newForms;

    for (unsigned int outputDim = 0; outputDim < nbOutputDims; ++outputDim)
    {
        vector<FormType> modelForms;
        for (auto const& model : candidateModels)
        {
            modelForms.push_back(model->getForms()[outputDim]);
        }

        unsigned int const totalNbDimensions = modelForms[0].usedDimensions.getTotalNbDimensions();

        auto fIt = modelForms.begin();
        auto const fEnd = modelForms.end();

        //        UsedDimensions formerlyNeededDimensions = UsedDimensions::allDimensions(totalNbDimensions);
        UsedDimensions formerlyNeededDimensions = fIt->neededDimensions;
        UsedDimensions relevantDimensions = fIt->relevantDimensions;
        unsigned int minPossibleComplexity = fIt->complexity;
        unsigned int maxAllowedComplexity = fIt->complexity;
        for(++fIt; fIt != fEnd; ++fIt)
        {
            formerlyNeededDimensions += fIt->neededDimensions;
            relevantDimensions += fIt->relevantDimensions;
            maxAllowedComplexity += fIt->complexity;
            minPossibleComplexity = max(minPossibleComplexity, fIt->complexity);
        }

        if (!shouldWork.empty())
        {
            maxAllowedComplexity = shouldWork[outputDim].first;
            formerlyNeededDimensions += shouldWork[outputDim].second;
        }

        UsedDimensions irrelevantDimensions = relevantDimensions.complement();
        UsedDimensions relevantOtherDimensions = relevantDimensions ^ formerlyNeededDimensions.complement();

        unsigned int const nbRelevantOtherDimensions = relevantOtherDimensions.getNbUsed();
        unsigned int const nbUnusedDimensions = formerlyNeededDimensions.getNbUnused();

        // The lowest complexity form which has been found
        list<FormType> bestForms;
        unsigned int bestComplexity = maxAllowedComplexity + 2;

        // We look for a form that would fit the points,
        //   and use a binary search to get the lowest complexity one
        // We don't immediately test the highest complexity
        //   because the computational cost can be prohibitive
        for (unsigned int nbAdditionalDimensions = 0;
             nbAdditionalDimensions <= nbUnusedDimensions; ++nbAdditionalDimensions)
        {
            unsigned int lowerBound = minPossibleComplexity;
            unsigned int upperBound = maxAllowedComplexity;

            // Instead of trying one complexity in [lowerBound, upperBound], we will consider a range,
            //   so that we don't miss forms with lower complexities but different dimensions.
            // That way, if no form in the range fits, we know the complexity was not high enough
            unsigned int middleComplexityRangeMin,  middleComplexityRangeMax;

            while (upperBound >= lowerBound)
            {
                if (upperBound - lowerBound < 5)
                {
                    // We finish in one go
                    middleComplexityRangeMax = upperBound;
                    middleComplexityRangeMin = lowerBound;
                }
                else
                {
                    if (bestComplexity < maxAllowedComplexity)
                    {
                        // At least one success, we go into the binary search normally
                        middleComplexityRangeMax = (lowerBound + upperBound) / 2;
                    }
                    else
                    {
                        // We don't take (lowerBound + upperBound) / 2
                        //  because higher complexities cost more to try
                        middleComplexityRangeMax = (3*lowerBound + upperBound) / 4;
                    }

                    if (lowerBound == 1)
                    {
                        middleComplexityRangeMin = 1;
                    }
                    else
                    {
                        middleComplexityRangeMin = max(ApproximatorType::getComplexityRangeLowerBound(
                                                           totalNbDimensions, middleComplexityRangeMax),
                                                       lowerBound);
                    }

                }

                // We look for possible forms with complexity in [middleComplexityRangeMin, middleComplexityRangeMax]
                //   which use exactly nbAdditionalDimensions dimensions not in formerlyNeededDimensions,
                //   with a priority on additional dimensions from relevantOtherDimensions
                list<list<FormType>> possibleForms;
                if (nbAdditionalDimensions <= nbRelevantOtherDimensions)
                {
                    possibleForms = ApproximatorType::getFormsInComplexityRange(
                                formerlyNeededDimensions,
                                relevantOtherDimensions,
                                nbAdditionalDimensions,
                                middleComplexityRangeMin,
                                middleComplexityRangeMax);
                }
                else
                {
                    possibleForms = ApproximatorType::getFormsInComplexityRange(
                                relevantDimensions,
                                irrelevantDimensions,
                                nbAdditionalDimensions - nbRelevantOtherDimensions,
                                middleComplexityRangeMin,
                                middleComplexityRangeMax);
                }

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
                            bool const fitResult = ApproximatorType::tryFit(
                                        form, nbPoints, mBegin, mEnd, outputID, outputDim);
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

        for (auto& form : bestForms)
        {
            bestNewForm.neededDimensions ^= form.usedDimensions;    //intersection
            bestNewForm.relevantDimensions += form.usedDimensions;  // union
        }

        newForms.push_back(bestNewForm);
    }

    static_pointer_cast<NodeType>(node)->setForms(newForms);
    return node;
}


// Tries have one model "steal" points (or rather, submodels) from each other:
// - candidate0 and candidate1 are models containing only one point (leaves)
// - the top-level model containing candidate0 tries to steal candidate1
//    or a model that contains it, and vice-versa
// - we go with the option which leads to the lowest sum of complexities
// => if point stealing was possible: we return true and the new models
//    otherwise: we return false and an empty list
template <typename ApproximatorType>
std::pair<bool, std::list<std::shared_ptr<Models::Model<ApproximatorType>>>>
ModelList<ApproximatorType>::pointStealing(
        Models::ModelIterator<ApproximatorType> candidate0,
        Models::ModelIterator<ApproximatorType> candidate1)
{
    using std::vector;
    using std::list;
    using std::map;
    using std::pair;
    using std::make_pair;
    using std::shared_ptr;
    using std::static_pointer_cast;
    using std::sort;
    using ModelType = Models::Model<ApproximatorType>;
    using NodeType = Models::Node<ApproximatorType>;
    using ModelPointer = shared_ptr<ModelType>;
    //    using ModelIterator = Models::ModelIterator<ApproximatorType>;

    auto smartTryMerge = [this](
            vector<ModelPointer> candidateModels)
            -> ModelPointer
    {
        // Store merge result and use that info whenever possible to avoid computation
        static map<vector<ModelPointer>, ModelPointer> alreadyComputed;
        sort(candidateModels.begin(), candidateModels.end());
        if (alreadyComputed.count(candidateModels) > 0)
        {
            return alreadyComputed.at(candidateModels);
        }
        else
        {
            auto const mergeResult = tryMerge(candidateModels);
            alreadyComputed.emplace(candidateModels, mergeResult);
            return mergeResult;
        }
    };

    auto getTreeWithoutSubmodel = [this, &smartTryMerge](list<ModelPointer> treeBranch) -> ModelPointer
    {
        // - treeBranch is the list containing a model, then its child,
        //    then this child's child, etc,
        //    leading to a submodel that we want to remove
        // - what we remove has to be a submodel,
        //    which means that treeBranch should contain at least 2 elements
        vector<ModelPointer> orphans;

        // We determine all orphan submodels
        // The variable named 'removed' contains the removed submodel at first,
        //  and contains the top-level model when the loop ends
        auto removed = treeBranch.back();
        treeBranch.pop_back();

        do
        {
            auto const parent = treeBranch.back();
            for (auto const& child : static_pointer_cast<NodeType>(parent)->getSubmodels())
            {
                if (child != removed)
                {
                    orphans.push_back(child);
                }
            }
            removed = parent;
            treeBranch.pop_back();
        } while (!treeBranch.empty());

        // If there is only one, it is now a top-level model and we return it
        if (orphans.size() == 1)
        {
            return orphans.back();
        }

        // Otherwise we fuse all orphans into one model
        //  (which has at worst the same form that the former top-level model)
        vector<pair<unsigned int, UsedDimensions>> topLevelModelCharacteristics;
        for (auto const& form : removed->getForms())
        {
            topLevelModelCharacteristics.push_back(make_pair(form.complexity, form.usedDimensions));
        }

        auto const newModel = tryMerge(orphans, topLevelModelCharacteristics);

        if (!newModel)
        {
            throw std::runtime_error("Orphan submodels could not be merged back into one");
        }

        return newModel;
    };


    // We initialize some stuff
    vector<list<ModelPointer>> treeBranches{candidate0.getTreeBranch(), candidate1.getTreeBranch()};
    vector<ModelPointer> const topModels{treeBranches[0].front(), treeBranches[1].front()};
    vector<unsigned int> topModelComplexities(2, 0);

    for (unsigned int i = 0; i < 2; ++i)
    {
        if (topModels[i]->isLeaf())
        {
            topModelComplexities[i] = 1;
        }
        else
        {
            for (auto const& form : static_pointer_cast<NodeType>(topModels[i])->getForms())
            {
                topModelComplexities[i] += form.complexity;
            }
        }
    }

    unsigned int bestComplexity = topModelComplexities[0] + topModelComplexities[1];
    list<ModelPointer> bestModels;

    // We try to merge the top-level models
    {
        auto const newModel = smartTryMerge(topModels);
        if (newModel)
        {
            unsigned int complexity = 0;
            for (auto const& form : static_pointer_cast<NodeType>(newModel)->getForms())
            {
                complexity += form.complexity;
            }

            if (complexity <= bestComplexity)
            {
                bestComplexity = complexity;
                bestModels = {newModel};
            }
        }
    }

    // We try to find if a merge between a top-level model and a submodel could be better
    for (unsigned int i = 0; i < 2; ++i)
    {
        auto& treeBranch = treeBranches[i];
        auto const& otherBranchTopModel = topModels[1-i];

        auto subModel = treeBranch.back();
        treeBranch.pop_back();
        while (!treeBranch.empty())
        {
            auto const newModel = smartTryMerge(vector<ModelPointer>{otherBranchTopModel, subModel});
            if (newModel)
            {
                unsigned int complexity = 0;
                for (auto const& form : static_pointer_cast<NodeType>(newModel)->getForms())
                {
                    complexity += form.complexity;
                }

                treeBranch.push_back(subModel);
                auto const restOfTheBranch = getTreeWithoutSubmodel(treeBranch);
                treeBranch.pop_back();

                if (restOfTheBranch->isLeaf())
                {
                    complexity += 1;
                }
                else
                {
                    for (auto const& form : static_pointer_cast<NodeType>(restOfTheBranch)->getForms())
                    {
                        complexity += form.complexity;
                    }
                }

                if (complexity < bestComplexity)
                {
                    bestComplexity = complexity;
                    bestModels = {newModel, restOfTheBranch};
                }
            }
            subModel = treeBranch.back();
            treeBranch.pop_back();
        }
    }

    if (bestModels.empty())
    {
        // No merge succeeded
        return {false, {}};
    }
    else
    {
        return {true, bestModels};
    }
}


// Merge models as much as possible by having them "steal" points (or rather, submodels) from each other,
//  starting with the closest pair of submodels
template <typename ApproximatorType>
void ModelList<ApproximatorType>::performPointStealing()
{
    using std::vector;
    using std::priority_queue;
    using std::map;
    using std::pair;
    using std::make_pair;
    using std::shared_ptr;
    using ModelDistance = Models::ModelDistance;
    using ModelType = Models::Model<ApproximatorType>;
    using ModelPointer = shared_ptr<ModelType>;
    using ModelPointerPair = pair<ModelPointer, ModelPointer>;
    using ModelIterator = Models::ModelIterator<ApproximatorType>;

    // We compute all distances between pairs of points,
    //  and map each point to an iterator
    priority_queue<pair<ModelDistance, ModelPointerPair>,
            vector<pair<ModelDistance, ModelPointerPair>>,
            HasGreaterDistance<ModelPointerPair>> distances;
    map<shared_ptr<ModelType>, ModelIterator> pointToIterator;
    {
        auto const mEnd = models.end();
        for (auto mIt0 = models.begin(); mIt0 != mEnd; ++mIt0)
        {
            auto const pEnd0 = Models::pointsEnd<ApproximatorType>(*mIt0);
            for (auto pIt0 = Models::pointsBegin<ApproximatorType>(*mIt0); pIt0 != pEnd0; ++pIt0)
            {
                pointToIterator.emplace(pIt0.getTreeBranch().back(), pIt0);
            }

            auto mIt1 = mIt0; ++mIt1;
            for (; mIt1 != mEnd; ++mIt1)
            {
                auto const pEnd1 = Models::pointsEnd<ApproximatorType>(*mIt1);
                for (auto pIt0 = Models::pointsBegin<ApproximatorType>(*mIt0); pIt0 != pEnd0; ++pIt0)
                {
                    for (auto pIt1 = Models::pointsBegin<ApproximatorType>(*mIt1); pIt1 != pEnd1; ++pIt1)
                    {
                        distances.push(
                                    make_pair(
                                        ModelDistance(*pIt0, *pIt1, outputID),
                                        make_pair(pIt0.getTreeBranch().back(),
                                                  pIt1.getTreeBranch().back()
                                                  )
                                        )
                                    );
                    }
                }
            }
        }
    }

    // Now we consider all pairs of points, one by one
    while (!distances.empty())
    {
        // We retrieve the pair from the distance priority queue
        auto currentPair = distances.top().second;
        distances.pop();
        auto const pointIterator0 = pointToIterator.at(currentPair.first);
        auto const pointIterator1 = pointToIterator.at(currentPair.second);

        // We skip cases where both points belong to the same model
        if (pointIterator0.getTreeBranch().front() == pointIterator1.getTreeBranch().front())
        {
            continue;
        }

        // We call pointStealing() with the points
        //  and update the models and the map if it succeeds
        auto pointStealingResult = pointStealing(pointIterator0, pointIterator1);
        if (pointStealingResult.first)
        {
            // We remove the old models
            {
                auto const toRemove0 = pointIterator0.getTreeBranch().front();
                auto const toRemove1 = pointIterator1.getTreeBranch().front();

                removeModel(toRemove0);
                removeModel(toRemove1);
            }

            // We add the new models and update the map
            {
                for (auto const& newModel : pointStealingResult.second)
                {
                    addModel(newModel);
                    auto const pEnd = Models::pointsEnd<ApproximatorType>(newModel);
                    for (auto pIt = Models::pointsBegin<ApproximatorType>(newModel);
                         pIt != pEnd; ++pIt)
                    {
                        auto const point = pIt.getTreeBranch().back();
                        pointToIterator.erase(point);
                        pointToIterator.emplace(point, pIt);
                    }
                }
            }
        }
    }
}



}
