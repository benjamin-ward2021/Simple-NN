#pragma once
#include <vector>
#include "Layer.hpp"

class Network {
public:
	Network(std::vector<int> topology);
	void train(int epochs, int randomSeed, double learningRate, double momentumFactor, std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> targets);
	void createCsv(std::string filename, std::vector<std::vector<double>> inputs);
	void print();
private:
	void forwardPass(std::vector<double> inputs);
	void backwardPass(std::vector<double> targets);
	void updateWeights(double learningRate, double momentumFactor);
	// Mean squared error
	static double mse(double target, double output);
	static double mse(std::vector<double> target, std::vector<double> output);
	static double mseDerivative(double target, double output);
	std::vector<double> getOutputs();
	std::vector<Layer> layers;
};