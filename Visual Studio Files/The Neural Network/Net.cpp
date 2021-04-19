#include <Net.h>
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
			std::cout << "Made a new Neuron!" << std::endl;
		}

		//Force the bias node's output value to 1.0. IT's the last neuron created above
		m_layers.back().back().setOutputVal(1.0);
	}
}

void Net::feedForward(const std::vector<double>& inputVals)
{
	assert(inputVals.size() == m_layers[0].size() - 1);

	for (unsigned i = 0; i < inputVals.size(); ++i) {
		m_layers[0][i].setOutputVal(i);
	}

	// Forward propagate
	for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
		Layer& prevLayer = m_layers[layerNum - 1];
		for ( unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n){
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
		m_error += delta*delta; //Error will be the sum of squares of error
	}
	
	m_error /= outputLayer.size() - 1; //Get average error squared
	std::cout << m_error << std::endl;
	m_error = sqrt(m_error); //Sqrt to get RMS




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


