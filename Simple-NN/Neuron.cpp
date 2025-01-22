#include <iostream>
#include <random>
#include "Neuron.hpp"
#include "Layer.hpp"

Neuron::Neuron(int index, int numWeights) {
	this->index = index;
	val = 0;
	output = 0;
	delta = 0;
	// Set random weights between 0 and 1
	weights.reserve(numWeights);
	for (int i = 0; i < numWeights; i++) {
		weights.push_back(getRandom());
	}
}

void Neuron::forwardPass(const Layer &prevLayer) {
	val = 0;
	for (int i = 0; i < prevLayer.neurons.size(); i++) {
		val += prevLayer.neurons[i].output * prevLayer.neurons[i].weights[index];
	}

	output = sigmoid(val);
}

void Neuron::backwardPassOutputLayer(double target) {
	double error = target - output;
	delta = error * sigmoidDerivative(val);
}

void Neuron::backwardPassHiddenLayer(const Layer &nextLayer) {
	double error = 0.0;
	// We didn't contribute to the error of the bias since it has no incoming connections
	for (int i = 0; i < nextLayer.neurons.size() - 1; i++) {
		error += nextLayer.neurons[i].delta * weights[i];
	}

	delta = error * sigmoidDerivative(val);
}

void Neuron::updateWeights(double learningRate, Layer &prevLayer) {
	// Update our incoming connections
	for (int i = 0; i < prevLayer.neurons.size(); i++) {
		prevLayer.neurons[i].weights[index] += learningRate * delta * prevLayer.neurons[i].output;
	}
}

void Neuron::setOutput(double output) {
	this->output = output;
}

void Neuron::print() {
	std::cout << "Neuron " << index << " has val: " << val << std::endl;
	std::cout << "Neuron " << index << " has activatedVal: " << output << std::endl;
}

// sigmoid = 1 / (1 + exp(-x))
double Neuron::sigmoid(double val) {
	return 1.0 / (1.0 + exp(-val));
}
// derivative of sigmoid = sigmoid(x) * (1 - sigmoid(x))
double Neuron::sigmoidDerivative(double output) {
	double sig = sigmoid(output);
	return sig * (1 - sig);
}

double Neuron::getRandom() {
	// Technically this isn't a double precision random number since rand() returns an int, but that's ok
	return rand() / double(RAND_MAX);
}