#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <Eigen/Dense>

class TrainingData
{
    public:
        TrainingData(const std::string filename);
        bool isEof(void) {return m_trainingDataFile.eof();}
        std::pair<unsigned int, unsigned int> getData(std::vector<double> &in, std::vector<double> &out);
    private:
        std::ifstream m_trainingDataFile;
};

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


struct Connection
{
    double weight;
    double deltaWeight;
};

class Neuron;
typedef std::vector<Neuron> Layer;

class Neuron
{
    public:
        Neuron(unsigned int numOutputs, unsigned int myIndex);
        void setOutputVal(double val) {m_outputVal = val;}
        double getOutputVal(void) const {return m_outputVal;}
        void feedForward(const Layer &prevLayer);
        void calcOutputGradients(double targetVal);
        void calcHiddenGradients(const Layer &nextLayer);
        void updateInputWeights(Layer &prevLayer);
		std::vector<Connection> m_outputWeights;
    private:
        static double eta;
        static double alpha;
        static double randomWeight(void) { return rand() / double(RAND_MAX);}
        double m_outputVal;
        unsigned int m_myIndex;
        static double transferFunction(double x);
        static double transferFunctionDerivative(double x);
        double m_gradient;
        double sumDOW(const Layer &nextLayer) const;
};

double Neuron::eta = 0.35;
double Neuron::alpha = 0.5;

Neuron::Neuron(unsigned int numOutputs, unsigned int myIndex)
{
	for (unsigned c = 0; c < numOutputs; ++c) {
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
	}

	m_myIndex = myIndex;
}

void Neuron::feedForward(const Layer& prevLayer)
{
	double sum = 0.0;

	for (unsigned n = 0; n < prevLayer.size(); ++n) {
		sum += prevLayer[n].getOutputVal() * 
			prevLayer[n].m_outputWeights[m_myIndex].weight;
	}

	m_outputVal = transferFunction(sum);
}

void Neuron::calcOutputGradients(double targetVal)
{
	double delta = targetVal - m_outputVal;
	m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);

}

void Neuron::calcHiddenGradients(const Layer& nextLayer)
{
	double dow = sumDOW(nextLayer);
	m_gradient = dow * transferFunctionDerivative(m_outputVal);
}

void Neuron::updateInputWeights(Layer &prevLayer)
{
	for (unsigned n = 0; n < prevLayer.size(); ++n) {
		Neuron& neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

		double newDeltaWeight =
			// Individual input magnified by the gradient and training rate (Eta)
			eta
			* neuron.getOutputVal()
			* m_gradient
			//Also add momentum = a fraction of the previous delta weight
			+ alpha
			* oldDeltaWeight;

		neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
	}
}

double Neuron::transferFunction(double x)
{
	return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
	return (1.0-x*x);
}

double Neuron::sumDOW(const Layer& nextLayer) const
{
	double sum = 0.0;

	// Sum our contributions of the errors at the nodes we feed

	for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}
	return sum;
}

//Class Net
class Net 
{
public:
    Net(const std::vector<unsigned int> &topology);
    void feedForward(const std::vector<double> &inputVals);
    void backProp(const std::vector<double> &targetVals);
    void getResults(std::vector<double> &resultVals) const;
    double getRecentAverageError(int pass) const;
    void saveWeights();
    void loadWeights();
    void viewWeights();
    std::vector<double> predict(std::vector<double> &in);
private:
    std::vector<Layer> m_layers;
    double m_error;
    double m_recentAverageError;
    double m_recentAverageSmoothingFactor;

};

double Net::getRecentAverageError(int pass) const
{
    return m_recentAverageError / double(pass);
}

std::vector<double> Net::predict(std::vector<double> &in)
{
	Net::feedForward(in);
    std::vector<double> output;
    Net::getResults(output);
    return output;
}

void Net::saveWeights()
{
    std::ofstream out;
    out.open("weights.csv");
    for(int i = 0; i < m_layers.size()-1; i++) //for each layer except last one
    {
        for(int j = 0; j < m_layers[i].size(); j++) //for each neuron in each layer
        {
			for (int k = 0; k < m_layers[i + 1].size()-1; k++) 
			{
				out << m_layers[i][j].m_outputWeights[k].weight;
				if (k < m_layers[i+1].size() - 2)
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

    for(int i = 0; i < m_layers.size()-1; i++) //for each layer except last one
    {
        for(int j = 0; j < m_layers[i].size(); j++) //for each neuron in each layer
        {
			std::getline(in, line);
			std::stringstream ss(line);
			for (int k = 0; k < m_layers[i + 1].size()-1; k++) 
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
    for(int i = 0; i < m_layers.size(); i++)
    {
        for(int j = 0; j < m_layers[i].size(); j++)
        {
            std::cout<<m_layers[i][j].getOutputVal()<<" ";
        }
        std::cout<<std::endl;
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

void showVectorVals(std::string label, std::vector<double> &v)
{
    std::cout<<label<<" ";
    for(unsigned int i = 0; i < v.size(); i++)
    {
        std::cout<<v[i]<<" ";
    }
    std::cout<<std::endl;
}

void train()
{
    TrainingData trainData("data.csv");
    std::vector<unsigned int> topology;
    topology.push_back(4);
    topology.push_back(5);
    topology.push_back(5);
    topology.push_back(3);
    Net myNet(topology);
    std::ifstream weights;
    weights.open("weights.csv");
    if(weights)
        myNet.loadWeights();

    std::vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;

    while (!trainData.isEof())
    {
        trainingPass++;
		
        if(trainData.getData(inputVals, targetVals).first != topology[0])
            break;
        
        //showVectorVals("Values: ", inputVals);
        //showVectorVals("Targets: ", targetVals);

        myNet.feedForward(inputVals);
        myNet.getResults(resultVals);

        assert(targetVals.size() == topology.back());

        myNet.backProp(targetVals);
        //myNet.viewWeights();
		if (trainingPass % 1000 == 0) {
			std::cout << std::endl << "Pass " << trainingPass;
			std::cout << "\tError: " << myNet.getRecentAverageError(trainingPass) << std::endl;
		}
        
    }
        myNet.saveWeights();
        std::cout<<std::endl<<"Done!"<<std::endl;
}

void linReg()
{


}
void predict()
{
	int inputSex;
	int inputAge;
	int inputRace;
	int inputMed;
	while (true)
	{
		//system("CLS");
		std::cout << "*********************************************************" << std::endl;
		std::cout << "Covid Survival Model" << std::endl;
		std::cout << "*********************************************************" << std::endl;
		std::cout << "Enter Sex: " << std::endl;
		std::cout << "(1) Female" << std::endl;
		std::cout << "(2) Male" << std::endl;
		std::cout << "(3) Other" << std::endl;
		std::cout << "(4) Not applicable" << std::endl << ">";
		std::cin >> inputSex;

		if (inputSex > 0 && inputSex < 5)
			break;
		else
			std::cout << "Invalid input, try again..." << std::endl;
	}

	while (true)
	{
		std::cout << "*********************************************************" << std::endl;
		std::cout << "Enter Age Group: " << std::endl;
		std::cout << "(1) 0-9 Years" << std::endl;
		std::cout << "(2) 10 - 19 Years" << std::endl;
		std::cout << "(3) 20 - 29 Years" << std::endl;
		std::cout << "(4) 30 - 39 Years" << std::endl;
		std::cout << "(5) 40 - 49 Years" << std::endl;
		std::cout << "(6) 50 - 59 Years" << std::endl;
		std::cout << "(7) 60 - 69 Years" << std::endl;
		std::cout << "(8) 70 - 79 Years" << std::endl;
		std::cout << "(9) 80+ Years" << std::endl;
		std::cout << "(10) N/A" << std::endl << ">";
		std::cin >> inputAge;
		if (inputAge > 0 && inputAge < 11)
			break;
		else
			std::cout << "Invalid input, try again..." << std::endl;
	}

	while (true)
	{
		std::cout << "*********************************************************" << std::endl;
		std::cout << "Enter Race:" << std::endl;
		std::cout << "(1) American Indian/Alaska Native (Non-Hispanic)" << std::endl;
		std::cout << "(2) Asian (Non-Hispanic)" << std::endl;
		std::cout << "(3) Black (Non-Hispanic)" << std::endl;
		std::cout << "(4) Hispanic/Latino" << std::endl;
		std::cout << "(5) Native Hawaiian/Other Pacific Islander (Non-Hispanic)" << std::endl;
		std::cout << "(6) White (Non-Hispanic)" << std::endl;
		std::cout << "(7) Multiple/Other (Non-Hispanic)" << std::endl;
		std::cout << "(8) N/A" << std::endl << ">";
		std::cin >> inputRace;
		if (inputRace > 0 && inputRace < 9)
			break;
		else
			std::cout << "Invalid input, try again..." << std::endl;
	}

	while (true)
	{
		std::cout << "*********************************************************" << std::endl;
		std::cout << "Pre-Existing Medical Conditions?" << std::endl;
		std::cout << "(1) Yes" << std::endl;
		std::cout << "(2) No" << std::endl << ">";
		std::cin >> inputMed;
		if (inputMed > 0 && inputMed < 3)
			break;
		else
			std::cout << "Invalid input, try again..." << std::endl;
	}

	std::cout << "*********************************************************" << std::endl;
	double numDiffSex = 4.0;
	double numDiffAge = 10.0;
	double numDiffRace = 8.0;

	double outSex = inputSex / numDiffSex;
	double outAge = inputAge / numDiffAge;
	double outRace = inputRace / numDiffRace;
	double outMed = inputMed - 1;

	std::vector<unsigned int> topology;
	topology.push_back(4);
	topology.push_back(5);
	topology.push_back(5);
	topology.push_back(3);
	Net myNet(topology);
	myNet.loadWeights();
	std::vector<double> send = { outSex, outAge, outRace, outMed };
	std::vector<double> stats = myNet.predict(send);
	myNet.viewWeights();
	std::cout << "Hospitalization chance: " << stats[0] * 100 << "%\tICU chance: " << stats[1] * 100 << "%\tDeath chance: " << stats[2] * 100 << "%" << std::endl;

}
void nnPicker()
{
    int in = 0;
    while (true)
    {
        std::cout << "*********************************************************" << std::endl;
        std::cout << "Enter mode: " << std::endl;
        std::cout << "(1) Train Neural Network" << std::endl;
        std::cout << "(2) Predict Using NN" << std::endl;

        std::cin >> in;

		if (in == 1)
			train();
		else
			predict();
    }
    
}

int main()
{
    while(true)
    {
        int input;
        std::cout<<"Mode select:"<<std::endl<<"(1) Neural Network"<<std::endl<<"(2) Linear Regression"<<std::endl<<"> "; //Nice touch with the arrow here
        std::cin >> input;

        switch (input)
        {
        case 1:
            nnPicker();
            break;
        case 2:
            linReg();
        default:
            break;
        }
    }
    
}