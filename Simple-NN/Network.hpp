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
	void print();
private:
	std::vector<Layer> layers;
};