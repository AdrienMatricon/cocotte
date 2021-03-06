#ifndef COCOTTE_DISTANCE_H
#define COCOTTE_DISTANCE_H


#include <vector>
#include <cocotte/datatypes.h>



namespace Cocotte {



namespace DistanceType
{
enum Type
{
    L_1_XT,
    L_2_SQUARED_XT,
    CLOSEST_DIM_X,
    RANDOM
};
}



template <DistanceType::Type>
class Distance {};



////////////////
//// L_1_XT ////
////////////////
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
    friend bool operator<(Distance<DistanceType::L_1_XT> const& lhs,
                          Distance<DistanceType::L_1_XT> const& rhs);

    friend bool operator>(Distance<DistanceType::L_1_XT> const& lhs,
                          Distance<DistanceType::L_1_XT> const& rhs)
    {return rhs < lhs;}

    friend bool operator<=(Distance<DistanceType::L_1_XT> const& lhs,
                          Distance<DistanceType::L_1_XT> const& rhs)
    {return !(rhs < lhs);}

    friend bool operator>=(Distance<DistanceType::L_1_XT> const& lhs,
                          Distance<DistanceType::L_1_XT> const& rhs)
    {return !(lhs < rhs);}

    // Returns a distance X such that smaller < X
    static Distance<DistanceType::L_1_XT> getBiggerDistanceThan(Distance const& smaller);
};



////////////////////////
//// L_2_SQUARED_XT ////
////////////////////////
template <>
class Distance<DistanceType::L_2_SQUARED_XT>
{

private:
    double value;

public:
    // Constructor
    Distance(Cocotte::DataPoint const& lhs, Cocotte::DataPoint const& rhs, unsigned int outputID);

    // Default constructor/copy constructor/copy operator
    Distance() = default;
    Distance(Distance<DistanceType::L_2_SQUARED_XT> const&) = default;
    Distance<DistanceType::L_2_SQUARED_XT>& operator=(Distance<DistanceType::L_2_SQUARED_XT> const&) = default;

    // Relational operator
    friend bool operator<(Distance<DistanceType::L_2_SQUARED_XT> const& lhs,
                          Distance<DistanceType::L_2_SQUARED_XT> const& rhs);

    friend bool operator>(Distance<DistanceType::L_2_SQUARED_XT> const& lhs,
                          Distance<DistanceType::L_2_SQUARED_XT> const& rhs)
    {return rhs < lhs;}

    friend bool operator<=(Distance<DistanceType::L_2_SQUARED_XT> const& lhs,
                          Distance<DistanceType::L_2_SQUARED_XT> const& rhs)
    {return !(rhs < lhs);}

    friend bool operator>=(Distance<DistanceType::L_2_SQUARED_XT> const& lhs,
                          Distance<DistanceType::L_2_SQUARED_XT> const& rhs)
    {return !(lhs < rhs);}

    // Returns a distance X such that smaller < X
    static Distance<DistanceType::L_2_SQUARED_XT> getBiggerDistanceThan(Distance const& smaller);
};



////////////////////////
//// CLOSEST_DIM_X /////
////////////////////////
template <>
class Distance<DistanceType::CLOSEST_DIM_X>
{

private:
    std::vector<Measure> coordinateDifferences;
    std::vector<unsigned int> smallestDifferences;

public:
    // Constructor
    Distance(Cocotte::DataPoint const& lhs, Cocotte::DataPoint const& rhs, unsigned int outputID);

    // Default constructor/copy constructor/copy operator
    Distance() = default;
    Distance(Distance<DistanceType::CLOSEST_DIM_X> const&) = default;
    Distance<DistanceType::CLOSEST_DIM_X>& operator=(Distance<DistanceType::CLOSEST_DIM_X> const&) = default;

    // Relational operator
    friend bool operator<(Distance<DistanceType::CLOSEST_DIM_X> const& lhs,
                          Distance<DistanceType::CLOSEST_DIM_X> const& rhs);

    friend bool operator>(Distance<DistanceType::CLOSEST_DIM_X> const& lhs,
                          Distance<DistanceType::CLOSEST_DIM_X> const& rhs)
    {return rhs < lhs;}

    friend bool operator<=(Distance<DistanceType::CLOSEST_DIM_X> const& lhs,
                          Distance<DistanceType::CLOSEST_DIM_X> const& rhs)
    {return !(rhs < lhs);}

    friend bool operator>=(Distance<DistanceType::CLOSEST_DIM_X> const& lhs,
                          Distance<DistanceType::CLOSEST_DIM_X> const& rhs)
    {return !(lhs < rhs);}

    // Returns a distance X such that smaller < X
    static Distance<DistanceType::CLOSEST_DIM_X> getBiggerDistanceThan(Distance const& smaller);
};



////////////////////////
//////// RANDOM ////////
////////////////////////
template <>
class Distance<DistanceType::RANDOM>
{
private:
    double value;

public:
    // Constructor
    Distance(Cocotte::DataPoint const& lhs, Cocotte::DataPoint const& rhs, unsigned int outputID);

    // Default constructor/copy constructor/copy operator
    Distance() = default;
    Distance(Distance<DistanceType::RANDOM> const&) = default;
    Distance<DistanceType::RANDOM>& operator=(Distance<DistanceType::RANDOM> const&) = default;

    // Relational operator
    friend bool operator<(Distance<DistanceType::RANDOM> const& lhs,
                          Distance<DistanceType::RANDOM> const& rhs);

    friend bool operator>(Distance<DistanceType::RANDOM> const& lhs,
                          Distance<DistanceType::RANDOM> const& rhs)
    {return rhs < lhs;}

    friend bool operator<=(Distance<DistanceType::RANDOM> const& lhs,
                          Distance<DistanceType::RANDOM> const& rhs)
    {return !(rhs < lhs);}

    friend bool operator>=(Distance<DistanceType::RANDOM> const& lhs,
                          Distance<DistanceType::RANDOM> const& rhs)
    {return !(lhs < rhs);}

    // Returns a distance X such that smaller < X
    static Distance<DistanceType::RANDOM> getBiggerDistanceThan(Distance const& smaller);
};



}



#endif
