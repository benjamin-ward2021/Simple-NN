#include <iostream>
#include "Network.hpp"

int main()
{
	std::vector<int> topology = { 2,3,1 };
	Network nn(topology);

	std::vector<std::vector<double>> sampleInputs = { {0,0},{0,1},{1,0},{1,1} };
	std::vector<std::vector<double>> sampleTargets = { {0},{1},{1},{0} };

	nn.train(300000, 0.01, sampleInputs, sampleTargets);

	std::cout << "Expected: 0" << std::endl;
	nn.forwardPass(sampleInputs[0]);
	nn.print();

	std::cout << "Expected: 1" << std::endl;
	nn.forwardPass(sampleInputs[1]);
	nn.print();

	std::cout << "Expected: 1" << std::endl;
	nn.forwardPass(sampleInputs[2]);
	nn.print();

	std::cout << "Expected: 0" << std::endl;
	nn.forwardPass(sampleInputs[3]);
	nn.print();

	std::cout << "Curveball" << std::endl;
	nn.forwardPass({0.8,0.2});
	nn.print();
}