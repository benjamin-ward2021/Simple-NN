#include <iostream>
#include "Network.hpp"

// TODO: Autogenerate inputs and targets; autogenerate CSV of loss over time; add ReLU;
//		 add "printLoss" for Network; add momentum
int main()
{
	std::vector<int> topology = { 2,3,1 };
	Network nn(topology);

	std::vector<std::vector<double>> sampleInputs = { {0,0},{0,1},{1,0},{1,1} };
	std::vector<std::vector<double>> sampleTargets = { {0},{1},{1},{0} };

	nn.train(300000, 0.01, sampleInputs, sampleTargets);

	std::cout << "Input: 0,0. Expected: 0" << std::endl;
	nn.forwardPass(sampleInputs[0]);
	nn.print();

	std::cout << "Input: 0,1. Expected: 1" << std::endl;
	nn.forwardPass(sampleInputs[1]);
	nn.print();

	std::cout << "Input: 1,0. Expected: 1" << std::endl;
	nn.forwardPass(sampleInputs[2]);
	nn.print();

	std::cout << "Input: 1,1. Expected: 0" << std::endl;
	nn.forwardPass(sampleInputs[3]);
	nn.print();

	std::cout << "Input: 0.5,0.5. Expected: 0" << std::endl;
	nn.forwardPass({0.5,0.5});
	nn.print();
}