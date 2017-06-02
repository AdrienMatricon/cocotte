
#include <list>
using std::list;
#include <vector>
using std::vector;
#include <utility>
using std::pair;
#include <iostream>
using std::cout;
using std::endl;
#include <cocotte/useddimensions.h>

namespace Cocotte {



// Constructors
UsedDimensions::UsedDimensions(unsigned int totalNb, list<unsigned int> ids):
    nbUsed(ids.size()),
    totalNbDimensions(totalNb),
    dimensionsIDs(ids)
{}



// Main functions
list<unsigned int> const& UsedDimensions::getIds() const
{
    return dimensionsIDs;
}


unsigned int UsedDimensions::getTotalNbDimensions() const
{
    return totalNbDimensions;
}


unsigned int UsedDimensions::getNbUsed() const
{
    return nbUsed;
}


unsigned int UsedDimensions::getNbUnused() const
{
    return totalNbDimensions - nbUsed;
}


// Returns a UsedDimensions using every dimensions
UsedDimensions UsedDimensions::allDimensions(unsigned int totalNbDimensions)
{
    list<unsigned int> IDs;
    for (unsigned int i = 0u; i < totalNbDimensions; ++i)
    {
        IDs.push_back(i);
    }
    return UsedDimensions(totalNbDimensions, IDs);
}


// Returns a UsedDimensions using only the unused dimensions
UsedDimensions UsedDimensions::complement() const
{
    list<unsigned int> IDs;

    auto dIter = dimensionsIDs.begin();
    auto const dEnd = dimensionsIDs.end();

    unsigned int i = 0;

    // We iterate through the list and add all missing dimensions to the list
    if (dIter != dEnd)
    {
        unsigned int next = *dIter;
        do
        {
            if (i == next)
            {
                next = *(++dIter);
            }
            else
            {
                IDs.push_back(i);
            }

            ++i;

        } while (dIter != dEnd);
    }

    // We add all remaining dimensions
    for (; i < totalNbDimensions; ++i)
    {
        IDs.push_back(i);
    }

    return UsedDimensions(totalNbDimensions, IDs);
}


// Returns a list of combinations of d used dimensions
list<UsedDimensions> UsedDimensions::getCombinations(unsigned int d) const
{
    if (d > nbUsed)
    {
        // No combination works
        return list<UsedDimensions>{};
    }
    else if (d < 1)
    {
        // Only one combination (nothing) works
        return list<UsedDimensions>(1, UsedDimensions(totalNbDimensions));
    }

    // We convert the list to a vector to access it more easily
    vector<unsigned int> const asVector(dimensionsIDs.begin(), dimensionsIDs.end());


    // We create a vector of pairs to generate the combination
    // Each pair contains a list of dimensions of increasing indices in asVector,
    // and the indice of the next dimension in asVector
    vector<pair<list<unsigned int>,unsigned int>> combinations(
                1, pair<list<unsigned int>,unsigned int>(list<unsigned int>{}, 0));

    for (unsigned int currentNb = 0; currentNb < d; ++currentNb)
    {
        vector<pair<list<unsigned int>, unsigned int>> next;

        unsigned int const stillNeeded = d - currentNb;
        unsigned int const lastIdAddable = nbUsed - stillNeeded;
        unsigned int const endLoop = lastIdAddable + 1;

        for (auto const& comb : combinations)
        {
            for (unsigned int i = std::get<1>(comb); i < endLoop; ++i)
            {
                list<unsigned int> temp = std::get<0>(comb);
                temp.push_back(asVector[i]);
                next.push_back(pair<list<unsigned int>, unsigned int>(temp, i+1));
            }
        }

        combinations = std::move(next);
    }

    list<UsedDimensions> result;

    for (auto const& comb:combinations)
    {
        result.push_back(UsedDimensions(totalNbDimensions, std::get<0>(comb)));
    }

    return result;
}


// Same but with combinations of d-k used dimensions and k unused one
list<UsedDimensions> UsedDimensions::getCombinationsWithKUnused(unsigned int d, unsigned int k) const
{
    if (k < 1)
    {
        return getCombinations(d);
    }

    if (k == d)
    {
        return complement().getCombinations(k);
    }

    if ( (d > nbUsed + k) || (d > totalNbDimensions) )
    {
        // No combination works
        return list<UsedDimensions>{};
    }

    auto const partialResult = getCombinations(d-k);
    auto const newDimensionsCombinations = complement().getCombinations(k);

    list<UsedDimensions> result;

    for (auto const& usedDims : partialResult)
        for (auto& newDims : newDimensionsCombinations)
    {
        result.push_back(usedDims + newDims);
    }

    return result;
}



// Operators


// Union
UsedDimensions const& UsedDimensions::operator+=(UsedDimensions otherDimensions)
{
    list<unsigned int> newDimensionsIDs;

    // We iterate over both lists to find common dimension IDs
    auto dIter0 = dimensionsIDs.begin(),
            dIter1 = otherDimensions.dimensionsIDs.begin();
    auto const dEnd0 = dimensionsIDs.end(),
            dEnd1 = otherDimensions.dimensionsIDs.end();

    // If one of the list is empty, the intersection is empty
    while ((dIter0 != dEnd0) && (dIter1 != dEnd1))
    {
        auto const dim0 = *dIter0, dim1 = *dIter1;
        if (dim0 < dim1)
        {
            newDimensionsIDs.push_back(dim0);
            ++dIter0;
        }
        else if (dim0 > dim1)
        {
            newDimensionsIDs.push_back(dim1);
            ++dIter1;
        }
        else
        {
            newDimensionsIDs.push_back(dim0);
            ++dIter0; ++dIter1;
        }
    }

    if (dIter0 != dEnd0)
    {
        newDimensionsIDs.insert(newDimensionsIDs.end(), dIter0, dEnd0);
    }

    if (dIter1 != dEnd1)
    {
        newDimensionsIDs.insert(newDimensionsIDs.end(), dIter1, dEnd1);
    }

    *this = UsedDimensions(totalNbDimensions, newDimensionsIDs);
    return *this;
}


// Intersection
UsedDimensions const& UsedDimensions::operator^=(UsedDimensions otherDimensions)
{
    list<unsigned int> newDimensionsIDs;

    // We iterate over both lists to find common dimension IDs
    auto dIter0 = dimensionsIDs.begin(),
            dIter1 = otherDimensions.dimensionsIDs.begin();
    auto const dEnd0 = dimensionsIDs.end(),
            dEnd1 = otherDimensions.dimensionsIDs.end();

    // If one of the list is empty, the intersection is empty
    while ((dIter0 != dEnd0) && (dIter1 != dEnd1))
    {
        auto const dim0 = *dIter0, dim1 = *dIter1;
        if (dim0 < dim1)
        {
            ++dIter0;
        }
        else if (dim0 > dim1)
        {
            ++dIter1;
        }
        else
        {
            newDimensionsIDs.push_back(dim0);
            ++dIter0; ++dIter1;
        }
    }

    *this = UsedDimensions(totalNbDimensions, newDimensionsIDs);
    return *this;
}



}
