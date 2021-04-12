#pragma once
#include <vector>
#include <iostream>
#include <Neuron.h>
#include <assert.h>
class Net
{
private:
	std::vector<Layer> m_layers; 
	double m_error;
public:
	Net(const std::vector<unsigned> &topology);
	void feedForward(const std::vector<double>& inputVals);
	void backProp(const std::vector<double>& targetVals);
	void getResults(std::vector<double>& resultVals) const;
};

