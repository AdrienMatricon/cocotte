#ifndef COCOTTE_DISTANCE_H
#define COCOTTE_DISTANCE_H


#include <cocotte/datatypes.h>



namespace Cocotte {



namespace DistanceType
{
enum Type
{
    L_1_XT,
    CLOSEST_DIM_X
};
}



template <DistanceType::Type>
class Distance final
{};

template <>
class Distance<DistanceType::L_1_XT>
{

private:
    double value;

public:
    // Constructor
    Distance(Cocotte::DataPoint const& lhs, Cocotte::DataPoint const& rhs, unsigned int outputID);

    // Default constructor/copy constructor/copy operator
    Distance() = default;
    Distance(Distance<DistanceType::L_1_XT> const&) = default;
    Distance<DistanceType::L_1_XT>& operator=(Distance<DistanceType::L_1_XT> const&) = default;

    // Relational operator
    friend bool operator<(Distance const& lhs, Distance const& rhs);

    // Returns a distance X such that smaller < X
    static Distance<DistanceType::L_1_XT> getBiggerDistanceThan(Distance const& smaller);
};



}



#endif
