#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include "Neuron.h"
#include <string>
#include <sstream>
#include <cassert>

typedef std::vector<Neuron> Layer;

class Net
{
public:
	Net(const std::vector<unsigned int>& topology);
	void feedForward(const std::vector<double>& inputVals);
	void backProp(const std::vector<double>& targetVals);
	void getResults(std::vector<double>& resultVals) const;
	double getRecentAverageError(int pass) const;
	void saveWeights();
	void loadWeights();
	void viewWeights();
	std::vector<double> predict(std::vector<double>& in);
private:
	std::vector<Layer> m_layers;
	double m_error;
	double m_recentAverageError;
	double m_recentAverageSmoothingFactor;

};