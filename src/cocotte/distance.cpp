
#include <cmath>
using std::abs;
#include <cocotte/distance.h>

namespace Cocotte {



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


}
