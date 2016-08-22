#ifndef DATASOURCES_DATALOADER_H
#define DATASOURCES_DATALOADER_H


#include <string>
#include <vector>
#include <utility>
#include <datasources/datasource.h>


namespace DataSources {



class DataLoader final : public DataSource
{

private:

    std::vector<std::string> inputVariableNames;
    std::vector<std::vector<std::string>> outputVariableNames;

    std::vector<Cocotte::DataPoint> loadedData;
    std::vector<std::vector<double>> testDataInput;
    std::vector<std::vector<std::vector<double>>> testDataOutput;
    std::vector<std::vector<std::vector<double>>> testDataOutputPrecisions;


public:

    // Constructor
    // Default values to load training data,
    // add the variable names to load test data
    explicit DataLoader(std::string dataFileName,
                        std::string structureFileName,
                        std::vector<std::string> const& inputNames = {},
                        std::vector<std::vector<std::string>> const& outputNames={});

    // Main methods
    virtual Cocotte::DataPoint getTrainingDataPoint() override;
    virtual std::vector<Cocotte::DataPoint> getTrainingDataPoints(unsigned int nbDataPoints) override;
    virtual std::pair<std::vector<std::vector<double>>,
    std::pair<std::vector<std::vector<std::vector<double>>>,
    std::vector<std::vector<std::vector<double>>>>> getTestDataPoints(unsigned int nbPoints) override;
    virtual std::vector<std::string> getInputVariableNames() override;
    virtual std::vector<std::vector<std::string>> getOutputVariableNames() override;

private:
    // Helper methods
    std::vector<std::string> cutString(std::string line);
    bool isDouble(std::string item);

};



}



#endif
