#include <iostream>
#include <random>
#include "Neuron.hpp"
#include "Layer.hpp"

Layer::Layer(int index, int numNeurons, int numNeuronsNextLayer) {
	this->index = index;
	// We need an extra neuron for the bias
	for (int i = 0; i < numNeurons + 1; i++) {
		neurons.push_back(Neuron(i, numNeuronsNextLayer));
	}

	// The bias will always have an output of 1
	neurons.back().setOutput(1.0);
}

void Layer::forwardPass(const Layer &prevLayer) {
	// Don't need to do forward pass for bias, since it's always 1
	for (int i = 0; i < neurons.size() - 1; i++) {
		neurons[i].forwardPass(prevLayer);
	}
}

void Layer::backwardPassOutputLayer(const std::vector<double> &targets) {
	// The last bias doesn't contribute to anything, so we ignore it
	for (int i = 0; i < neurons.size() - 1; i++) {
		neurons[i].backwardPassOutputLayer(targets[i]);
	}
}
void Layer::backwardPassHiddenLayer(const Layer &nextLayer) {
	for (int i = 0; i < neurons.size(); i++) {
		neurons[i].backwardPassHiddenLayer(nextLayer);
	}
}

void Layer::updateWeights(double learningRate, Layer &prevLayer) {
	for (int i = 0; i < neurons.size() - 1; i++) {
		neurons[i].updateWeights(learningRate, prevLayer);
	}
}

void Layer::print() {
	for (int i = 0; i < neurons.size(); i++) {
		std::cout << "Layer " << index << ":" << std::endl;
		neurons[i].print();
	}

	std::cout << std::endl;
}