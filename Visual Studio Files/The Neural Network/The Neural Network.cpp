#include <iostream>
#include <vector>
#include <Net.h>

int main()
{
	std::vector<double> inputVals;
	std::vector<unsigned> topology;
	topology.push_back(3);
	topology.push_back(2);
	topology.push_back(1);
	Net myNet(topology);
}

