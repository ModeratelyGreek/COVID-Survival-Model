#include "Net.h"
double Net::getRecentAverageError(int pass) const
{
	return m_recentAverageError / double(pass);
}

std::vector<double> Net::predict(std::vector<double>& in)
{
	Net::feedForward(in);
	std::vector<double> output;
	Net::getResults(output);
	return output;
}

void Net::saveWeights()
{
	std::ofstream out;
	out.open("../Resource Files/weights.csv");
	for (int i = 0; i < m_layers.size() - 1; i++) //for each layer except last one
	{
		for (int j = 0; j < m_layers[i].size(); j++) //for each neuron in each layer
		{
			for (int k = 0; k < m_layers[i + 1].size() - 1; k++)
			{
				out << m_layers[i][j].m_outputWeights[k].weight;
				if (k < m_layers[i + 1].size() - 2)
					out << ",";
				else out << std::endl;
			}


		}
		out << std::endl;
	}
}

void Net::loadWeights()
{
	std::ifstream in;
	in.open("weights.csv");

	std::string line;
	std::string data;

	for (int i = 0; i < m_layers.size() - 1; i++) //for each layer except last one
	{
		for (int j = 0; j < m_layers[i].size(); j++) //for each neuron in each layer
		{
			std::getline(in, line);
			std::stringstream ss(line);
			for (int k = 0; k < m_layers[i + 1].size() - 1; k++)
			{
				getline(ss, data, ',');
				m_layers[i][j].m_outputWeights[k].weight = std::stod(data);
			}
		}
		std::getline(in, line);
	}
}

void Net::viewWeights()
{
	for (int i = 0; i < m_layers.size(); i++)
	{
		for (int j = 0; j < m_layers[i].size(); j++)
		{
			std::cout << m_layers[i][j].getOutputVal() << " ";
		}
		std::cout << std::endl;
	}
}

Net::Net(const std::vector<unsigned>& topology)
{
	unsigned numLayers = topology.size();
	for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
		m_layers.push_back(Layer());

		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
		/*The number of outputs for all neurons in this layer, if its the last layter, is 0,
		but if not the last layer is the number of neurons in the next layer*/

		//New Layer created, now filling it with neurons,
		//and adding a bias to neurons in the layer
		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
			m_layers.back().push_back(Neuron(numOutputs, neuronNum)); //m_layers.back() is the last layer.
			//We are pushing a new neuron onto the last layer topology[layerNum] times
		}

		//Force the bias node's output value to 1.0. IT's the last neuron created above
		m_layers.back().back().setOutputVal(1.0);
	}
}

void Net::feedForward(const std::vector<double>& inputVals)
{
	assert(inputVals.size() == m_layers[0].size() - 1);

	for (unsigned i = 0; i < inputVals.size(); ++i) {
		m_layers[0][i].setOutputVal(inputVals[i]);
	}

	// Forward propagate
	for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
		Layer& prevLayer = m_layers[layerNum - 1];
		for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
			m_layers[layerNum][n].feedForward(prevLayer);
		}
	}
}

void Net::backProp(const std::vector<double>& targetVals)
{
	//Calculate overall net error (RMS/Root Mean Square Error of output neuron errors)
	Layer& outputLayer = m_layers.back();
	double m_error = 0.0;

	for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error += delta * delta; //Error will be the sum of squares of error
	}

	m_error /= outputLayer.size() - 1; //Get average error squared
	m_error = sqrt(m_error); //Sqrt to get RMS
	m_recentAverageError += m_error;

	//Calculate output Layer Gradients
	for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	//Calculate hidden layer gradients

	for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
		Layer& hiddenLayer = m_layers[layerNum];
		Layer& nextLayer = m_layers[layerNum + 1];

		for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	//For all layers from outputs to the first hidden layer,
	//update all them connection weights

	for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
		Layer& layer = m_layers[layerNum];
		Layer& prevLayer = m_layers[layerNum - 1];

		for (unsigned n = 0; n < layer.size() - 1; ++n) {
			layer[n].updateInputWeights(prevLayer);
		}
	}
}

void Net::getResults(std::vector<double>& resultVals) const
{
	resultVals.clear();
	for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}

void showVectorVals(std::string label, std::vector<double>& v)
{
	std::cout << label << " ";
	for (unsigned int i = 0; i < v.size(); i++)
	{
		std::cout << v[i] << " ";
	}
	std::cout << std::endl;
}