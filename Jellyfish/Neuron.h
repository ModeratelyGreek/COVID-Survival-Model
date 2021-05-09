#pragma once
#include<vector>
class Neuron;
typedef std::vector<Neuron> Layer;

struct Connection
{
	double weight;
	double deltaWeight;
};

class Neuron
{
public:
	Neuron(unsigned int numOutputs, unsigned int myIndex);
	void setOutputVal(double val) { m_outputVal = val; }
	double getOutputVal(void) const { return m_outputVal; }
	void feedForward(const Layer& prevLayer);
	void calcOutputGradients(double targetVal);
	void calcHiddenGradients(const Layer& nextLayer);
	void updateInputWeights(Layer& prevLayer);
	std::vector<Connection> m_outputWeights;
private:
	static double eta;
	static double alpha;
	static double randomWeight(void) { return rand() / double(RAND_MAX); }
	double m_outputVal;
	unsigned int m_myIndex;
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	double m_gradient;
	double sumDOW(const Layer& nextLayer) const;
};