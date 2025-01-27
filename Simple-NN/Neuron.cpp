#include <iostream>
#include <random>
#include "Neuron.hpp"
#include "Layer.hpp"

Neuron::Neuron(int index, int numWeights) {
	this->index = index;
	val = 0;
	output = 0;
	gradient = 0;
	// Set random weights between 0 and 1
	weights.reserve(numWeights);
	prevWeightUpdate.reserve(numWeights);
	for (int i = 0; i < numWeights; i++) {
		weights.push_back(getRandom());
		prevWeightUpdate.push_back(0);
	}
}

void Neuron::forwardPass(const Layer &prevLayer) {
	val = 0;
	for (int i = 0; i < prevLayer.neurons.size(); i++) {
		val += prevLayer.neurons[i].output * prevLayer.neurons[i].weights[index];
	}

	output = relu(val);
}

void Neuron::backwardPassOutputLayer(double target, double (*lossDerivative)(double target, double output)) {
	double error = lossDerivative(target, output);
	gradient = error * reluDerivative(val);
}

void Neuron::backwardPassHiddenLayer(const Layer &nextLayer) {
	double error = 0.0;
	// We didn't contribute to the error of the bias since it has no incoming connections
	for (int i = 0; i < nextLayer.neurons.size() - 1; i++) {
		error += nextLayer.neurons[i].gradient * weights[i];
	}

	gradient = error * reluDerivative(val);
}

// Note that this implementation requires calling updateWeights after each backpropogation, which is fine since we are
// using stochastic gradient descent and not minibatch / fullbatch gradient descent
void Neuron::updateWeights(double learningRate, double momentumFactor, Layer &prevLayer) {
	// Update our incoming connections
	for (int i = 0; i < prevLayer.neurons.size(); i++) {
		double weightUpdate = -learningRate * gradient * prevLayer.neurons[i].output + momentumFactor * prevLayer.neurons[i].prevWeightUpdate[index];
		prevLayer.neurons[i].weights[index] += weightUpdate;
		prevLayer.neurons[i].prevWeightUpdate[index] = weightUpdate;
	}
}

void Neuron::setOutput(double output) {
	this->output = output;
}

double Neuron::getOutput() {
	return output;
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

double Neuron::relu(double input) {
	return std::max(0.0, input);
}
double Neuron::reluDerivative(double input) {
	return input > 0 ? 1 : 0;
}

double Neuron::getRandom() {
	// Technically this isn't a double precision random number since rand() returns an int, but that's ok
	return rand() / double(RAND_MAX);
}