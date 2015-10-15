#include "SelfOptimizingUnit.h"

#include <algorithm>

#include <iostream>

using namespace deep;

void SelfOptimizingUnit::createRandom(int numStates, int numActions, int numCells, float initMinWeight, float initMaxWeight, float initMinInhibition, float initMaxInhibition, std::mt19937 &generator) {
	std::uniform_real_distribution<float> weightDist(initMinWeight, initMaxWeight);
	std::uniform_real_distribution<float> inhibitionDist(initMinInhibition, initMaxInhibition);

	_inputs.assign(numStates, 0.0f);
	_reconstruction.assign(_inputs.size(), 0.0f);

	_cells.resize(numCells);

	_qConnections.resize(_cells.size());

	for (int i = 0; i < numCells; i++) {
		_cells[i]._gateFeedForwardConnections.resize(_inputs.size());

		_cells[i]._gateBias._weight = weightDist(generator);

		for (int j = 0; j < _inputs.size(); j++)
			_cells[i]._gateFeedForwardConnections[j]._weight = weightDist(generator);

		_cells[i]._gateLateralConnections.resize(_cells.size());

		for (int j = 0; j < _cells.size(); j++)
			_cells[i]._gateLateralConnections[j]._weight = inhibitionDist(generator);

		_qConnections[i]._weight = weightDist(generator);
	}

	_actions.resize(numActions);

	for (int i = 0; i < _actions.size(); i++) {
		_actions[i]._connections.resize(_cells.size());

		for (int j = 0; j < _cells.size(); j++)
			_actions[i]._connections[j]._weight = weightDist(generator);
	}
}

void SelfOptimizingUnit::simStep(float reward, float sparsity, float gamma, float gateFeedForwardAlpha, float gateLateralAlpha, float gateBiasAlpha, float qAlpha, float actionAlpha, float gammaLambda, float explorationStdDev, float explorationBreak, std::mt19937 &generator) {
	for (int i = 0; i < _cells.size(); i++) {
		float activation = 0.0f;

		for (int j = 0; j < _inputs.size(); j++)
			activation += _cells[i]._gateFeedForwardConnections[j]._weight * _inputs[j];

		_cells[i]._gateActivation = activation;
	}

	float q = 0.0f;

	// Inhibit
	for (int i = 0; i < _cells.size(); i++) {
		float inhibition = _cells[i]._gateBias._weight;

		for (int j = 0; j < _cells.size(); j++)
			inhibition += (_cells[i]._gateActivation > _cells[j]._gateActivation ? 1.0f : 0.0f) * _cells[i]._gateLateralConnections[j]._weight;

		_cells[i]._gate = inhibition < 1.0f ? 1.0f : 0.0f;

		q += _qConnections[i]._weight * _cells[i]._gate;
	}

	float tdError = reward + gamma * q - _prevValue;
	float qAlphaTdError = qAlpha * tdError;

	_prevValue = q;

	// Reconstruct
	for (int i = 0; i < _reconstruction.size(); i++) {
		float recon = 0.0f;
		float div = 0.0f;

		for (int j = 0; j < _cells.size(); j++) {
			recon += _cells[j]._gateFeedForwardConnections[i]._weight * _cells[j]._gate;

			div += _cells[j]._gate;
		}

		_reconstruction[i] = recon;
	}

	float sparsitySquared = sparsity * sparsity;

	for (int i = 0; i < _cells.size(); i++) {
		// Learn SDRs
		if (_cells[i]._gate > 0.0f) {
			for (int j = 0; j < _inputs.size(); j++)
				_cells[i]._gateFeedForwardConnections[j]._weight += gateFeedForwardAlpha * (_inputs[j] - _reconstruction[j]);
		}

		for (int j = 0; j < _cells.size(); j++)
			_cells[i]._gateLateralConnections[j]._weight = std::max(0.0f, _cells[i]._gateLateralConnections[j]._weight + gateLateralAlpha * (_cells[i]._gate * _cells[j]._gate - sparsitySquared)); //(_cells[i]._gateActivation > _cells[j]._gateActivation ? 1.0f : 0.0f)

		_cells[i]._gateBias._weight += gateBiasAlpha * (_cells[i]._gate - sparsity);

		// Learn Q
		_qConnections[i]._weight += qAlphaTdError * _qConnections[i]._trace;

		_qConnections[i]._trace = std::max(_qConnections[i]._trace * gammaLambda, _cells[i]._gate);
	}

	// Optimize actions
	//float actionAlphaTdError = actionAlpha * tdError;
	for (int i = 0; i < _actions.size(); i++) {
		float delta = tdError * (_actions[i]._exploratoryState - _actions[i]._state);

		// Update actions base on previous state
		for (int j = 0; j < _cells.size(); j++) {		
			_actions[i]._connections[j]._trace = _actions[i]._connections[j]._trace * gammaLambda + delta * _cells[j]._gatePrev;

			// Trace order update reverse here on purpose since action is based on previous state
			_actions[i]._connections[j]._weight += actionAlpha * _actions[i]._connections[j]._trace;
		}
	}

	// Find new actions
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::normal_distribution<float> pertDist(0.0f, explorationStdDev);

	for (int i = 0; i < _actions.size(); i++) {
		float activation = 0.0f;

		for (int j = 0; j < _cells.size(); j++)
			activation += _actions[i]._connections[j]._weight * _cells[j]._gate;

		_actions[i]._state = sigmoid(activation);

		if (dist01(generator) < explorationBreak)
			_actions[i]._exploratoryState = dist01(generator);
		else
			_actions[i]._exploratoryState = std::min(1.0f, std::max(0.0f, _actions[i]._state + pertDist(generator)));
	}

	// Buffer update
	for (int i = 0; i < _cells.size(); i++)
		_cells[i]._gatePrev = _cells[i]._gate;
}