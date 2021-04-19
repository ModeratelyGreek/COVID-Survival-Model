#include <iostream>
#include <vector>
#include <Net.h>

int main()
{
	std::vector<double> inputVals;
	std::vector<unsigned> topology;
	topology.push_back(10);
	topology.push_back(10);
	topology.push_back(10);
	Net myNet(topology);
}

