#ifndef DATASOURCES_DATASOURCE_H
#define DATASOURCES_DATASOURCE_H


#include <vector>
#include <string>
#include <utility>
#include<cocotte/datatypes.h>



namespace DataSources {



struct TestData
{
    std::vector<std::vector<double>> xValues;
    std::vector<std::vector<std::vector<double>>> tValues;
    std::vector<std::vector<std::vector<double>>> tPrecisions;
};



class DataSource
{

public:
    virtual Cocotte::DataPoint getTrainingDataPoint() = 0;
    virtual std::vector<Cocotte::DataPoint> getTrainingDataPoints(unsigned int nbPoints) = 0;
    virtual TestData getTestDataPoints(unsigned int nbPoints) = 0;
    virtual std::vector<std::string> getInputVariableNames() = 0;
    virtual std::vector<std::vector<std::string>> getOutputVariableNames() = 0;
    virtual unsigned int getNbDataPoints() = 0;

};



}



#endif
