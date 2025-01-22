#pragma once
#include <vector>
#include "Neuron.hpp"

class Layer {
public:
	Layer(int index, int numNeurons, int numNeuronsNextLayer);
	void forwardPass(const Layer &prevLayer);
	void backwardPassOutputLayer(const std::vector<double> &targets);
	void backwardPassHiddenLayer(const Layer &nextLayer);
	void updateWeights(double learningRate, Layer &prevLayer);
	void print();
	std::vector<Neuron> neurons;
private:
	int index;
};