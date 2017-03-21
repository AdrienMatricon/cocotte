
#include <cmath>
using std::abs;
#include <algorithm>
using std::sort;
#include <cocotte/distance.h>

namespace Cocotte {



////////////////
//// L_1_XT ////
////////////////
// Constructor
Distance<DistanceType::L_1_XT>::Distance(Cocotte::DataPoint const& lhs, Cocotte::DataPoint const& rhs, unsigned int outputID) : value(0)
{
    for (auto it0 = lhs.x.begin(), end0 = lhs.x.end(), it1 = rhs.x.begin();
         it0 != end0; ++it0, ++it1)
    {
        value += abs(it0->value - it1->value);
    }

    for (auto it0 = lhs.t[outputID].begin(), end0 = lhs.t[outputID].end(), it1 = rhs.t[outputID].begin();
         it0 != end0; ++it0, ++it1)
    {
        value += abs(it0->value - it1->value);
    }
}


// Relational operator
bool operator<(Distance<DistanceType::L_1_XT> const& lhs, Distance<DistanceType::L_1_XT> const& rhs)
{
    return lhs.value < rhs.value;
}


// Returns a distance X such that smaller < X
Distance<DistanceType::L_1_XT> Distance<DistanceType::L_1_XT>::getBiggerDistanceThan(Distance const& smaller)
{
    Distance<DistanceType::L_1_XT> bigger = smaller;
    bigger.value = 2*bigger.value + 0.01;
    return bigger;
}



////////////////////////
//// L_2_SQUARED_XT ////
////////////////////////
// Constructor
Distance<DistanceType::L_2_SQUARED_XT>::Distance(Cocotte::DataPoint const& lhs, Cocotte::DataPoint const& rhs, unsigned int outputID) : value(0)
{
    for (auto it0 = lhs.x.begin(), end0 = lhs.x.end(), it1 = rhs.x.begin();
         it0 != end0; ++it0, ++it1)
    {
        double const temp = it0->value - it1->value;
        value += temp * temp;
    }

    for (auto it0 = lhs.t[outputID].begin(), end0 = lhs.t[outputID].end(), it1 = rhs.t[outputID].begin();
         it0 != end0; ++it0, ++it1)
    {
        double const temp = it0->value - it1->value;
        value += temp * temp;
    }
}


// Relational operator
bool operator<(Distance<DistanceType::L_2_SQUARED_XT> const& lhs, Distance<DistanceType::L_2_SQUARED_XT> const& rhs)
{
    return lhs.value < rhs.value;
}


// Returns a distance X such that smaller < X
Distance<DistanceType::L_2_SQUARED_XT> Distance<DistanceType::L_2_SQUARED_XT>::getBiggerDistanceThan(Distance const& smaller)
{
    Distance<DistanceType::L_2_SQUARED_XT> bigger = smaller;
    bigger.value = 2*bigger.value + 0.01;
    return bigger;
}



////////////////////////
//// CLOSEST_DIM_X /////
////////////////////////
// Constructor
Distance<DistanceType::CLOSEST_DIM_X>::Distance(Cocotte::DataPoint const& lhs, Cocotte::DataPoint const& rhs, unsigned int outputID)
{
    (void) outputID;

    // We compute the difference on each coordinate of the input,
    // as well as the precision on this difference
    coordinateDifferences.reserve(lhs.x.size());
    smallestDifferences.reserve(lhs.x.size());

    unsigned int ID = 0u;
    for (auto it0 = lhs.x.begin(), end0 = lhs.x.end(), it1 = rhs.x.begin();
         it0 != end0; ++it0, ++it1, ++ID)
    {
        Measure difference{abs(it0->value - it1->value), it0->precision + it1->precision};
        coordinateDifferences.push_back(difference);
        smallestDifferences.push_back(ID);
    }

    // We sort the dimensions'IDs by difference
    sort(smallestDifferences.begin(), smallestDifferences.end(),
         [&](unsigned int left, unsigned int right){
        return coordinateDifferences[left].value < coordinateDifferences[right].value;
    });
}


// Relational operator
bool operator<(Distance<DistanceType::CLOSEST_DIM_X> const& lhs, Distance<DistanceType::CLOSEST_DIM_X> const& rhs)
{
    // If the smallest difference on a coordinate is significantly smaller for a distance
    // than the same difference for the other distance,
    // then this distance is smaller
    // Otherwise we go on to the next smallest difference
    auto it0 = lhs.smallestDifferences.begin(), it1 = rhs.smallestDifferences.begin();
    auto const end0 = lhs.smallestDifferences.end(), end1 = rhs.smallestDifferences.end();

    auto& diff0 = lhs.coordinateDifferences;
    auto& diff1 = rhs.coordinateDifferences;

    while ((it0 != end0) && (it1 != end1))
    {
        auto const ID0 = *it0, ID1 = *it1;

        if (diff0[ID0].value < diff1[ID1].value)
        {
            if ( (diff0[ID0].value - diff1[ID0].value) < (diff0[ID0].precision + diff1[ID0].precision) )
            {
                return true;
            }

            ++it0;
        }
        else
        {
            if ( (diff1[ID1].value - diff0[ID1].value) < (diff0[ID1].precision + diff1[ID1].precision) )
            {
                return false;
            }

            ++it1;
        }
    }

    if (it0 != end0)
    {
        do
        {
            auto const ID0 = *it0;
            if ( (diff0[ID0].value - diff1[ID0].value) < (diff0[ID0].precision + diff1[ID0].precision) )
            {
                return true;
            }

            ++it0;
        } while  (it0 != end0);
    }

    return false;
}


// Returns a distance X such that smaller < X
Distance<DistanceType::CLOSEST_DIM_X> Distance<DistanceType::CLOSEST_DIM_X>::getBiggerDistanceThan(Distance const& smaller)
{
    Distance<DistanceType::CLOSEST_DIM_X> bigger = smaller;
    for (auto& difference : bigger.coordinateDifferences)
    {
        difference.value = 2*difference.value + 0.01;
    }

    return bigger;
}


}
