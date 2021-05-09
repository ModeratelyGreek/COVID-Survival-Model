#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

class TrainingData
{
    public:
        TrainingData(const std::string filename);
        bool isEof(void) {return m_trainingDataFile.eof();}
        std::pair<unsigned int, unsigned int> getData(std::vector<double> &in, std::vector<double> &out);
    private:
        std::ifstream m_trainingDataFile;
};