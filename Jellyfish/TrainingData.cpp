#include "TrainingData.h"

TrainingData::TrainingData(const std::string filename)
{
    m_trainingDataFile.open(filename.c_str());
    std::string temp;
    std::getline(m_trainingDataFile, temp);
}

std::pair<unsigned int, unsigned int> TrainingData::getData(std::vector<double> &in, std::vector<double> &out)
{
    in.clear();
    out.clear();

    std::string line;
    std::getline(m_trainingDataFile, line);
    std::stringstream ss(line);
    std::string data;

    unsigned int i = 0;
    while(std::getline(ss, data, ','))
    {
        if(i<4)
        {
            if(i == 0) getline(ss, data, ',');
            in.push_back(std::stod(data));
        }
        else
            out.push_back(std::stod(data));
        i++;
    }

    std::pair<unsigned int, unsigned int> output = std::make_pair(in.size(), out.size());
    return output;
}