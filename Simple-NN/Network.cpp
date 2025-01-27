#include <iostream>
#include <cassert>
#include <vector>
#include <random>
#include <fstream> // TODO: clean these up
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
	layers.back().backwardPassOutputLayer(targets, mseDerivative);
	for (int i = layers.size() - 2; i >= 0; i--) {
		layers[i].backwardPassHiddenLayer(layers[i + 1]);
	}
}

void Network::updateWeights(double learningRate, double momentumFactor) {
	for (int i = 1; i < layers.size(); i++) {
		layers[i].updateWeights(learningRate, momentumFactor, layers[i - 1]);
	}
}

// Uses online SGD
void Network::train(int epochs, int randomSeed, double learningRate, double momentumFactor, std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> targets) {
	std::default_random_engine randomEngine = std::default_random_engine(randomSeed);
	// This has both ends inclusive, so we need to subtract one to the max to get an index
	std::uniform_int_distribution<int> randomIndexGenerator(0, inputs.size() - 1);
	for (int i = 0; i < epochs; i++) {
		double loss = 0.0;
		for (int j = 0; j < inputs.size(); j++) {
			int randomIndex = randomIndexGenerator(randomEngine);
			forwardPass(inputs[randomIndex]);
			backwardPass(targets[randomIndex]);
			loss += mse(targets[randomIndex], getOutputs());
			updateWeights(learningRate, momentumFactor);
		}

		if (i % (epochs / 10) == 0 || i == epochs - 1) {
			std::cout << "Loss " << i << ": " << loss / inputs.size() << std::endl;
		}
	}
}

void Network::createCsv(std::string filename, std::vector<std::vector<double>> inputs) {
	std::ofstream csv;
	csv.open(filename + ".csv");
	csv << "x,predicted y" << std::endl;
	for (int i = 0; i < inputs.size(); i++) {
		forwardPass(inputs[i]);
		std::vector<double> outputs = getOutputs();
		// Currently we have only one input and one output dimension, so we can just get the 0th index of both
		csv << inputs[i][0] << "," << outputs[0] << std::endl;
	}

	csv.close();
}

// Not really much of a "mean" since there's only one error being calculated
double Network::mse(double target, double output) {
	return pow(output - target, 2);
}

// This is for multiple samples with one input and output parameter, not one sample with multiple parameters
double Network::mse(std::vector<double> target, std::vector<double> output) {
	assert(target.size() == output.size());
	double acc = 0.0;
	for (int i = 0; i < target.size(); i++) {
		double error = output[i] - target[i];
		acc += error * error;
	}

	return acc / target.size();
}

double Network::mseDerivative(double target, double output) {
	return 2 * (output - target);
}

void Network::print() {
	for (int i = 0; i < layers.size(); i++) {
		layers[i].print();
	}
}

std::vector<double> Network::getOutputs() {
	return layers.back().getOutputs();
}