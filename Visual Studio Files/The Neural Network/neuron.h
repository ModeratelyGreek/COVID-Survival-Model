#pragma once
#include <vector>
#include <cstdlib>
#include <cmath>
class Neuron;
typedef std::vector<Neuron> Layer;

struct Connection {
	double weight;
	double deltaWeight;
};
class Neuron
{
public:
	Neuron(unsigned numOutputs, unsigned myIndex);
	void setOutputVal(double val) { m_outputVal = val; };
	double getOutputVal() const { return m_outputVal; };
	void feedForward(const Layer &prevLayer);
	void calcOutputGradients(double targetVal);
	void calcHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);
private:
	static double randomWeight() { return rand() / double(RAND_MAX); }
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	double sumDOW(const Layer& nextLayer) const;
	double m_outputVal;
	unsigned m_myIndex;
	double m_gradient;
	static double eta; //[0.0...1.0] overall net training rate
	static double alpha; //[0.0...n] multiplier of last weight change (momentum)
	std::vector<Connection> m_outputWeights; //Adjacencylist
};





