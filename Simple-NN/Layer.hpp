#pragma once
#include <vector>
#include "Neuron.hpp"

class Layer {
public:
	Layer(int index, int numNeurons, int numNeuronsNextLayer);
	void forwardPass(const Layer &prevLayer);
	void backwardPassOutputLayer(const std::vector<double> &targets, double (*lossDerivative)(double target, double output));
	void backwardPassHiddenLayer(const Layer &nextLayer);
	void updateWeights(double learningRate, double momentumFactor, Layer &prevLayer);
	void print();
	std::vector<double> getOutputs();
	std::vector<Neuron> neurons;
private:
	int index;
};