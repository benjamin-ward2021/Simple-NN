#pragma once
#include <vector>

class Layer;

class Neuron {
public:
	Neuron(int index, int numWeights);
	void forwardPass(const Layer &prevLayer);
	void backwardPassOutputLayer(double target, double (*lossDerivative)(double target, double output));
	void backwardPassHiddenLayer(const Layer &nextLayer);
	void updateWeights(double learningRate, double momentumFactor, Layer &prevLayer);
	// Used for setting the input layer and biases
	void setOutput(double output);
	double getOutput();
	void print();
private:
	int index;
	double val;
	double output;
	double gradient;
	// One weight for each neuron in the next layer
	std::vector<double> weights;
	// This is the previous amount that we incremented each weight by; used for momentum
	std::vector<double> prevWeightUpdate;
	// Maybe activation functions should be passed via function pointer from Layer, instead of being in Neuron...
	double sigmoid(double input);
	double sigmoidDerivative(double input);
	double relu(double input);
	double reluDerivative(double input);
	double getRandom();
};