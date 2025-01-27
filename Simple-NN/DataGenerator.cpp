#include <iostream>
#include <vector>
#include "DataGenerator.hpp"

DataGenerator::DataGenerator(int randomSeed, double rangeMin, double rangeMax) {
	randomEngine = std::default_random_engine(randomSeed);
	uniformDistribution = std::uniform_real_distribution<double>(rangeMin, rangeMax);
}

void DataGenerator::generate(int numTrain, int numTest,
	std::vector<std::vector<double>> &trainInputs, std::vector<std::vector<double>> &trainTargets,
	std::vector<std::vector<double>> &testInputs, std::vector<std::vector<double>> &testTargets) {

	for (int i = 0; i < numTrain; i++) {
		double x = uniformDistribution(randomEngine);
		double y = f1(x);
		// We could have multiple inputs / outputs so we put them into vectors
		// ex. f(x1, x2) = 8*x1 + 13*x2 + 4
		std::vector<double> inputs = { x };
		std::vector<double> targets = { y };
		trainInputs.push_back(inputs);
		trainTargets.push_back(targets);
	}

	for (int i = 0; i < numTest; i++) {
		double x = uniformDistribution(randomEngine);
		double y = f1(x);
		std::vector<double> inputs = { x };
		std::vector<double> targets = { y };
		testInputs.push_back(inputs);
		testTargets.push_back(targets);
	}
}

// f1(x) = abs(0.1*x^3 - 7x) + 3
double DataGenerator::f1(double x) {
	return abs(0.1 * pow(x, 3) - 7 * x) + 3;
}