#pragma once
#include <vector>
#include "Layer.hpp"

class Network {
public:
	Network(std::vector<int> topology);
	void forwardPass(std::vector<double> inputs);
	void backwardPass(std::vector<double> targets);
	void updateWeights(double learningRate);
	void train(int epochs, double learningRate, std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> targets);
	// Mean squared error
	static double mse(double target, double output);
	static double mseDerivative(double target, double output);
	void print();
private:
	std::vector<Layer> layers;
};