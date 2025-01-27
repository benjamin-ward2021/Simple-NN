#include <iostream>
#include "Network.hpp"
#include "DataGenerator.hpp"

int main()
{
	DataGenerator generator(0, -10.0, 10.0);
	std::vector<std::vector<double>> trainInputs, trainTargets, testInputs, testTargets;
	generator.generate(5000, 2000, trainInputs, trainTargets, testInputs, testTargets);
	std::vector<int> topology = { 1,9,9,9,1 };
	Network nn(topology);
	nn.train(20000, 0, 0.000005, 0.5, trainInputs, trainTargets);
	nn.createCsv("example2", testInputs);
}