#ifndef DATASOURCES_DATASOURCE_H
#define DATASOURCES_DATASOURCE_H


#include <vector>
#include <string>
#include <utility>
#include<cocotte/datatypes.h>



namespace DataSources {



class DataSource
{

public:
    virtual Cocotte::DataPoint getTrainingDataPoint() = 0;
    virtual std::vector<Cocotte::DataPoint> getTrainingDataPoints(int nbPoints) = 0;
    virtual std::pair<std::vector<std::vector<double>>,
    std::pair<std::vector<std::vector<std::vector<double>>>,
    std::vector<std::vector<std::vector<double>>>>> getTestDataPoints(int nbPoints) = 0;
    virtual std::vector<std::string> getInputVariableNames() = 0;
    virtual std::vector<std::vector<std::string>> getOutputVariableNames() = 0;

};



}



#endif
