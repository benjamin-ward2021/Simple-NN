#include <iostream>
#include <cassert>
#include <vector>
#include "Layer.hpp"
#include "Network.hpp"

Network::Network(std::vector<int> topology) {
	// Set random seed for consistent results
	srand(42);

	// Needs to have at least an input and output layer
	assert(topology.size() > 1);
	// Initialize input and hidden layers
	for (int i = 0; i < topology.size() - 1; i++) {
		assert(topology[i] > 0);
		layers.push_back(Layer(i, topology[i], topology[i + 1]));
	}

	// Initialize output layer
	layers.push_back(Layer(topology.size() - 1, topology.back(), 0));
}

void Network::forwardPass(std::vector<double> inputs) {
	// One input per neuron, with one extra neuron acting as the bias
	assert(layers[0].neurons.size() - 1 == inputs.size());
	// Set the output of the first layer equal to the inputs
	for (int i = 0; i < layers[0].neurons.size() - 1; i++) {
		layers[0].neurons[i].setOutput(inputs[i]);
	}

	// Go through the hidden / output layers
	for (int i = 1; i < layers.size(); i++) {
		layers[i].forwardPass(layers[i - 1]);
	}
}

void Network::backwardPass(std::vector<double> targets) {
	layers.back().backwardPassOutputLayer(targets);
	for (int i = layers.size() - 2; i >= 0; i--) {
		layers[i].backwardPassHiddenLayer(layers[i + 1]);
	}
}

void Network::updateWeights(double learningRate) {
	for (int i = 1; i < layers.size(); i++) {
		layers[i].updateWeights(learningRate, layers[i - 1]);
	}
}

void Network::train(int epochs, double learningRate, std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> targets) {
	for (int i = 0; i < epochs; i++) {
		for (int j = 0; j < inputs.size(); j++) {
			forwardPass(inputs[j]);
			backwardPass(targets[j]);
			updateWeights(learningRate);
		}
	}
}

void Network::print() {
	for (int i = 0; i < layers.size(); i++) {
		layers[i].print();
	}
}