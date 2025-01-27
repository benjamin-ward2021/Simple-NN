#pragma once
#include <vector>
#include <random>

class DataGenerator {
public:
	DataGenerator(int randomSeed, double rangeMin, double rangeMax);
	const int inputDimension = 1;
	const int outputDimension = 1;

	void generate(int numTrain, int numTest,
		std::vector<std::vector<double>> &trainInputs, std::vector<std::vector<double>> &trainTargets,
		std::vector<std::vector<double>> &testInputs, std::vector<std::vector<double>> &testTargets);
private:
	double f1(double x);
	std::default_random_engine randomEngine;
	std::uniform_real_distribution<double> uniformDistribution;
};