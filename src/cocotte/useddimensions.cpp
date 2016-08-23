
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
UsedDimensions::UsedDimensions(unsigned int nbDimensions, list<unsigned int> dimensions):
    nbUsed(dimensions.size()),
    totalNbDimensions(nbDimensions),
    dimensionsIds(dimensions)
{}

UsedDimensions::UsedDimensions(unsigned int nbDimensions, list<unsigned int> dimensions, unsigned int used):
    nbUsed(used),
    totalNbDimensions(nbDimensions),
    dimensionsIds(dimensions)
{}



// Main functions
list<unsigned int> const& UsedDimensions::getIds() const
{
    return dimensionsIds;
}


unsigned int UsedDimensions::getTotalNbDimensions() const
{
    return totalNbDimensions;
}


unsigned int UsedDimensions::getNbUsed() const
{
    return nbUsed;
}


list<unsigned int> UsedDimensions::unusedDimensionsIds() const
{
    list<unsigned int> result;

    auto dIter = dimensionsIds.begin();
    auto const dEnd = dimensionsIds.end();

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
                result.push_back(i);
            }

            ++i;

        } while (dIter != dEnd);
    }

    // We add all remaining dimensions
    for (; i < totalNbDimensions; ++i)
    {
        result.push_back(i);
    }

    return result;
}


void UsedDimensions::addDimension(unsigned int id)
{
    auto dIter = dimensionsIds.begin();
    auto dEnd = dimensionsIds.end();

    // We go through the list until we reach its end or find a value above id
    while ((dIter != dEnd) && (*dIter < id))
    {
        ++dIter;
    }

    // If id is not already in the list, we add it
    if ((dIter == dEnd) || (*dIter > id))
    {
        dimensionsIds.insert(dIter, id);
        ++nbUsed;
    }
}


// Returns a list of combinations of d used dimensions
list<UsedDimensions> UsedDimensions::getCombinationsFromUsed(unsigned int d) const
{
    if (d > nbUsed)
    {
        // No combination works
        return list<UsedDimensions>{};
    }
    else if (d < 1)
    {
        // Only one combination (nothing) works
        return list<UsedDimensions>(1, UsedDimensions(totalNbDimensions, list<unsigned int>{}));
    }

    // We convert the list to a vector to access it more easily
    vector<unsigned int> const asVector(dimensionsIds.begin(), dimensionsIds.end());


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
        result.push_back(UsedDimensions(totalNbDimensions, std::get<0>(comb), d));
    }

    return result;
}


// Same but also with combinations of d-1 used dimensions and an unused one
list<UsedDimensions> UsedDimensions::getCombinationsFromUsedAndOne(unsigned int d) const
{
    if ( (d-1) > nbUsed )
    {
        // No combination works
        return list<UsedDimensions>{};
    }
    else if (d < 1)
    {
        // Only one combination (nothing) works
        return list<UsedDimensions>(1, UsedDimensions(totalNbDimensions, list<unsigned int>{}));
    }

    auto result = getCombinationsFromUsed(d);
    auto const partialResult = getCombinationsFromUsed(d-1);
    auto const complement = unusedDimensionsIds();


    for (auto& dim : complement)
    {
        auto pR = partialResult;
        for (auto& comb : pR)
        {
            comb.addDimension(dim);
        }
        result.insert(result.end(), pR.begin(), pR.end());
    }

    return result;
}



// Operators
UsedDimensions const& UsedDimensions::operator+=(UsedDimensions otherDimensions)
{
    auto dIter0 = dimensionsIds.begin(),
            dIter1 = otherDimensions.dimensionsIds.begin();
    auto const dEnd1 = otherDimensions.dimensionsIds.end();

    // If one of the list is empty, the union is equal to the other
    if (dIter0 == dimensionsIds.end())
    {
        dimensionsIds = otherDimensions.dimensionsIds;
        nbUsed = otherDimensions.nbUsed;
        return *this;
    }
    else if (dIter1 == dEnd1)
    {
        return *this;
    }

    // Otherwise, we iterate through both lists
    auto dim0 = *dIter0, dim1 = *dIter1;

    while (true)
    {
        if (dim0 > dim1)
        {
            // We iterate through the other list until we reach or pass dim0
            if(++dIter1 == dEnd1)
            {
                return *this;
            }
            dim1 = *dIter1;
        }
        else if (dim0 < dim1)
        {
            // If we passed dim0, we increment the iterator and to see if dim1 was between both values of dim0
            // If it is the case, dim1 is not in dimensionsIds and we insert it
            if (++dIter0 == dimensionsIds.end())
            {
                break;
            }

            dim0 = *dIter0;
            if (dim0 > dim1)
            {
                dimensionsIds.insert(dIter0, dim1);
                ++nbUsed;
                if(++dIter1 == dEnd1)
                {
                    return *this;
                }

                dim1 = *dIter1;
                --dIter0;
                dim0 = *dIter0;
            }
        }
        else
        {
            // If we simply reached dim0, we increment both iterators
            if(++dIter1 == dEnd1)
            {
                return *this;
            }

            if (++dIter0 == dimensionsIds.end())
            {
                break;
            }

            dim0 = *dIter0;
            dim1 = *dIter1;
        }
    }

    // We append the remaining dimensions to dimensionsIds
    dimensionsIds.insert(dIter0, dIter1, dEnd1);
    nbUsed = dimensionsIds.size();

    return *this;
}



}
