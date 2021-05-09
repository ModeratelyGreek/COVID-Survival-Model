#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <Eigen/Dense>
#include "TrainingData.h"
#include "Neuron.h"
#include "Net.h"


Eigen::MatrixXd s1(3, 1);
Eigen::MatrixXd s2(3, 1);
Eigen::MatrixXd s3(3, 1);
double error1, error2, error3;

void train()
{
    TrainingData trainData("../Resource Files/data.csv"); 
    std::vector<unsigned int> topology;
    topology.push_back(4);
    topology.push_back(5);
    topology.push_back(5);
	topology.push_back(5);
	topology.push_back(5);
    topology.push_back(3);
    Net myNet(topology);

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

void loadMatrices(Eigen::MatrixXd &A, Eigen::MatrixXd &out1, Eigen::MatrixXd &out2, Eigen::MatrixXd &out3)
{
	std::ifstream in;
	in.open("../Resource Files/data.csv");
	std::string data;
	std::getline(in, data, '\n'); //ditch the first line
	for (int i = 0; i < 642426; i++) //for each layer except last one
	{
		std::getline(in, data, ','); //ditch date double
		std::getline(in, data, ','); //sex
		A(i, 0) = stod(data);

		std::getline(in, data, ','); //age
		A(i, 1) = stod(data);

		std::getline(in, data, ','); //race
		A(i, 2) = stod(data);

		std::getline(in, data, ','); //med
		A(i, 3) = stod(data);

		std::getline(in, data, ','); //isHospitalized?
		out1(i, 0) = stod(data);

		std::getline(in, data, ','); //isICU?
		out2(i, 0) = stod(data);

		std::getline(in, data, ','); //isDead?
		out3(i, 0) = stod(data);

	}
}

void linReg()
{
	//Linear solve a with respect to b1 (multiple least squares regression for all inputs with respect to isHospitalized)
	std::cout << "Least Squares Multiple Regression with Respect to isHospitalized: " << std::endl;
	std::cout << "Least Squares Solution: " << s1 << std::endl;
	std::cout << "MSE: " << error1 << std::endl;
	std::cout << std::endl;
	
	//Linear solve a with respect to b2 (multiple least squares regression for all inputs with respect to isICU)
	std::cout << "Least Squares Multiple Regression with Respect to isICU: " << std::endl;
	std::cout << "Least Squares Solution: " << s2 << std::endl;
	std::cout << "MSE: " << error2 << std::endl;
	std::cout << std::endl;
	
	//Linear solve a with respect to b3 (multiple least squares regression for all inputs with respect to isDead?)
	std::cout << "Least Squares Multiple Regression with Respect to isDead?: " << std::endl;
	std::cout << "Least Squares Solution: " << s3 << std::endl;
	std::cout << "MSE: " << error3 << std::endl;
	std::cout << std::endl;
}

std::vector<double> calcLinearReg(double sex, double age, double race, double medCond)
{
	std::vector<double> solution;
	solution.push_back(s1(0, 0) * sex + //The solution for sex with respect to Hospitalization
		s1(1, 0) * age +				//The solution for age with respect to Hospitalization
		s1(2, 0) * race +				//The solution for race with respect to Hospitalization
		s1(3, 0) * medCond);			//The solution for med with respect to Hospitalization

	solution.push_back(s2(0, 0) * sex + //The solution for sex with respect to ICU
		s2(1, 0) * age +				//The solution for age with respect to ICU
		s2(2, 0) * race +				//The solution for race with respect to ICU
		s2(3, 0) * medCond);			//The solution for med with respect to ICU
		
	solution.push_back(s3(0, 0) * sex + //The solution for sex with respect to Death
		s3(1, 0) * age +				//The solution for age with respect to Death
		s3(2, 0) * race +				//The solution for race with respect to Death
		s3(3, 0) * medCond);			//The solution for med with respect to Death

	return solution;
} 

void predict()
{
	int inputSex;
	int inputAge;
	int inputRace;
	int inputMed;
	while (true)
	{
		system("CLS");
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
	std::vector<double> statsNN = myNet.predict(send);
	myNet.viewWeights();

	linReg();
	std::vector<double> statsLR = calcLinearReg(outSex, outAge, outRace, outMed);

	std::cout <<"Neural Network Prediction: \t"<<"Hospitalization chance: " << statsNN[0] * 100 << "%\tICU chance: " << statsNN[1] * 100 << "%\tDeath chance: " << statsNN[2] * 100 << "%" << std::endl;
	std::cout <<"Linear Regression Prediction: \t"<<"Hospitalization chance: " << statsLR[0] * 100 << "%\tICU chance: " << statsLR[1] * 100 << "%\tDeath chance: " << statsLR[2] * 100<<"%"<<std::endl;
	std::cout<< "Hospitalization Error: " <<error1<< "\tICU Error: " <<error2<< "\tDeath Error: " <<error3<<std::endl;

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
		else if (in ==2)
			predict();
		else break;
    }
    
}

int main()
{
	std::cout << "Please Wait while linear regression computes for dataset." << std::endl;
	//Calculate matrix stuff first:
	Eigen::MatrixXd a(642426, 4); //4 columns of input data: Gender, Age, Ethnicity, preExisting?
	Eigen::MatrixXd b1(642426, 1); //giant column of isHospitalized?
	Eigen::MatrixXd b2(642426, 1); //giant column of isICU?
	Eigen::MatrixXd b3(642426, 1); //giant column of isDead?

	loadMatrices(a, b1, b2, b3);

	s1 = a.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b1);
	error1 = (b1 - a * s1).col(0).squaredNorm() / 642426.0;
	system("CLS");
	std::cout << "Please Wait while linear regression computes for dataset.." << std::endl;
	s2 = a.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b2);
	error2 = (b2 - a * s2).col(0).squaredNorm() / 642426.0;
	system("CLS");
	std::cout << "Please Wait while linear regression computes for dataset..." << std::endl;
	s3 = a.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b3);
	error3 = (b3 - a * s3).col(0).squaredNorm() / 642426.0;
	system("CLS");
	std::cout << "Done!" << std::endl;
	std::cout << std::endl;
	std::cout << std::endl;

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