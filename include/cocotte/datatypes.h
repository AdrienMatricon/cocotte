#ifndef DATASOURCES_DATATYPES_H
#define DATASOURCES_DATATYPES_H


#include <vector>
#include <boost/serialization/vector.hpp>



namespace Cocotte {


struct Measure
{
    double value;
    double precision;

    // Serialization
    template<typename Archive>
    friend void serialize(Archive& archive, Measure& measure, const unsigned int version)
    {
        archive & measure.value;
        archive & measure.precision;
    }
};

struct DataPoint
{
    std::vector<Measure> x;
    std::vector<std::vector<Measure>> t;

    // Serialization
    template<typename Archive>
    friend void serialize(Archive& archive, DataPoint& point, const unsigned int version)
    {
        archive & point.x;
        archive & point.t;
    }
};



}



#endif
