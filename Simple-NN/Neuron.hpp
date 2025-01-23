#pragma once
#include <vector>

class Layer;

class Neuron {
public:
	Neuron(int index, int numWeights);
	void forwardPass(const Layer &prevLayer);
	void backwardPassOutputLayer(double target, double (*lossDerivative)(double target, double output));
	void backwardPassHiddenLayer(const Layer &nextLayer);
	void updateWeights(double learningRate, Layer &prevLayer);
	// Used for setting the input layer and biases
	void setOutput(double output);
	void print();
private:
	int index;
	double val;
	double output;
	double delta;
	// One weight for each neuron in the next layer
	std::vector<double> weights;
	// Maybe activation functions should be passed via function pointer from Layer, instead of being in Neuron...
	double sigmoid(double input);
	double sigmoidDerivative(double input);
	double getRandom();
};