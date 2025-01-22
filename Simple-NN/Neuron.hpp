#pragma once
#include <vector>

class Layer;

class Neuron {
public:
	Neuron(int index, int numWeights);
	void forwardPass(const Layer &prevLayer);
	void backwardPassOutputLayer(double target);
	void backwardPassHiddenLayer(const Layer &nextLayer);
	void updateWeights(double learningRate, Layer &prevLayer);
	// Used for setting the input layer
	void setOutput(double output);
	void print();
private:
	int index;
	double val;
	double output;
	double delta;
	// One weight for each neuron in the next layer
	std::vector<double> weights;
	double sigmoid(double input);
	double sigmoidDerivative(double input);
	double getRandom();
};