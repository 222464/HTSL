/*
AI Lib
Copyright (C) 2014 Eric Laukien

This software is provided 'as-is', without any express or implied
warranty.  In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
claim that you wrote the original software. If you use this software
in a product, an acknowledgment in the product documentation would be
appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#include "AutoEncoder.h"

using namespace deep;

void AutoEncoder::createRandom(size_t numInputs, size_t numOutputs, double minWeight, double maxWeight, std::mt19937 &generator) {
	_inputBiases.resize(numInputs);
	_hidden.resize(numOutputs);
	_inputErrorBuffer.resize(numInputs);

	std::uniform_real_distribution<double> distWeight(minWeight, maxWeight);

	for (size_t n = 0; n < _inputBiases.size(); n++)
		_inputBiases[n] = distWeight(generator);

	for (size_t n = 0; n < _hidden.size(); n++) {
		_hidden[n]._bias = distWeight(generator);

		_hidden[n]._weights.resize(numInputs);

		for (size_t w = 0; w < numInputs; w++)
			_hidden[n]._weights[w] = distWeight(generator);
	}
}

double AutoEncoder::crossoverChooseWeight(double w1, double w2, double averageChance, std::mt19937 &generator) {
	std::uniform_real_distribution<double> dist01(0.0f, 1.0f);

	if (dist01(generator) < averageChance)
		return (w1 + w2) * 0.5;

	return dist01(generator) < 0.5 ? w1 : w2;
}

void AutoEncoder::createFromParents(const AutoEncoder &parent1, const AutoEncoder &parent2, double averageChance, std::mt19937 &generator) {
	_inputBiases.resize(parent1._inputBiases.size());
	_hidden.resize(parent1._hidden.size());
	_inputErrorBuffer.resize(parent1._inputErrorBuffer.size());

	for (size_t n = 0; n < _inputBiases.size(); n++)
		_inputBiases[n] = crossoverChooseWeight(parent1._inputBiases[n], parent2._inputBiases[n], averageChance, generator);

	for (size_t n = 0; n < _hidden.size(); n++) {
		_hidden[n]._bias = crossoverChooseWeight(parent1._hidden[n]._bias, parent2._hidden[n]._bias, averageChance, generator);

		_hidden[n]._weights.resize(_inputBiases.size());

		for (size_t w = 0; w < _inputBiases.size(); w++)
			_hidden[n]._weights[w] = crossoverChooseWeight(parent1._hidden[n]._weights[w], parent2._hidden[n]._weights[w], averageChance, generator);
	}
}

void AutoEncoder::mutate(double perturbationChance, double perturbationStdDev, std::mt19937 &generator) {
	std::uniform_real_distribution<double> dist01(0.0, 1.0);
	std::normal_distribution<double> distPerturbation(0.0, perturbationStdDev);

	for (size_t n = 0; n < _inputBiases.size(); n++)
		_inputBiases[n] += dist01(generator) < perturbationChance ? distPerturbation(generator) : 0.0;

	for (size_t n = 0; n < _hidden.size(); n++) {
		_hidden[n]._bias += dist01(generator) < perturbationChance ? distPerturbation(generator) : 0.0;

		for (size_t w = 0; w < _inputBiases.size(); w++)
			_hidden[n]._weights[w] += dist01(generator) < perturbationChance ? distPerturbation(generator) : 0.0;
	}
}

void AutoEncoder::update(const std::vector<double> &inputs, std::vector<double> &outputs, double alpha) {
	if (outputs.size() != _hidden.size())
		outputs.resize(_hidden.size());

	for (size_t n = 0; n < _hidden.size(); n++) {
		double sum = _hidden[n]._bias;

		for (size_t w = 0; w < _hidden[n]._weights.size(); w++)
			sum += inputs[w] * _hidden[n]._weights[w];

		outputs[n] = sigmoid(sum);
	}

	if (alpha == 0.0)
		return;

	for (size_t n = 0; n < _inputBiases.size(); n++) {
		double sum = _inputBiases[n];

		for (size_t w = 0; w < _hidden.size(); w++)
			sum += outputs[w] * _hidden[w]._weights[n];

		double output = sigmoid(sum);

		_inputErrorBuffer[n] = (inputs[n] - sum);
	}

	for (size_t n = 0; n < _inputBiases.size(); n++)
		_inputBiases[n] += alpha * _inputErrorBuffer[n];

	for (size_t n = 0; n < _hidden.size(); n++) {
		double error = 0.0;

		for (size_t w = 0; w < _hidden[n]._weights.size(); w++)
			error += _inputErrorBuffer[w] * _hidden[n]._weights[w];

		error *= outputs[n] * (1.0 - outputs[n]);

		_hidden[n]._bias += alpha * error;

		for (size_t w = 0; w < _hidden[n]._weights.size(); w++) {
			_hidden[n]._weights[w] += alpha * (error * inputs[w] + _inputErrorBuffer[w] * outputs[n]);
		}
	}
}

void AutoEncoder::getReconstruction(const std::vector<double> &inputs, std::vector<double> &reconstruction) {
	std::vector<double> outputs(_hidden.size());

	for (size_t n = 0; n < _hidden.size(); n++) {
		double sum = _hidden[n]._bias;

		for (size_t w = 0; w < _hidden[n]._weights.size(); w++)
			sum += inputs[w] * _hidden[n]._weights[w];

		outputs[n] = sigmoid(sum);
	}

	if (reconstruction.size() != _inputBiases.size())
		reconstruction.resize(_inputBiases.size());

	for (size_t n = 0; n < _inputBiases.size(); n++) {
		double sum = _inputBiases[n];

		for (size_t w = 0; w < _hidden.size(); w++)
			sum += outputs[w] * _hidden[w]._weights[n];

		reconstruction[n] = sum;
	}
}

void AutoEncoder::reconstruction(const std::vector<double> &outputs, std::vector<double> &reconstruction) {
	if (reconstruction.size() != _inputBiases.size())
		reconstruction.resize(_inputBiases.size());

	for (size_t n = 0; n < _inputBiases.size(); n++) {
		double sum = _inputBiases[n];

		for (size_t w = 0; w < _hidden.size(); w++)
			sum += outputs[w] * _hidden[w]._weights[n];

		reconstruction[n] = sum;
	}
}