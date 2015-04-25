#include "SparseCoder.h"

#include <algorithm>

using namespace deep;

void SparseCoder::createRandom(int numVisible, int numHidden, float minWeight, float maxWeight, std::mt19937 &generator) {
	std::uniform_real_distribution<float> weightDistVisibleHidden(minWeight, maxWeight);
	std::uniform_real_distribution<float> weightDistHiddenHidden(0.0f, maxWeight);

	_visible.resize(numVisible);

	_hidden.resize(numHidden);

	for (int hi = 0; hi < numHidden; hi++) {
		_hidden[hi]._bias._weight = weightDistVisibleHidden(generator);

		_hidden[hi]._visibleHiddenConnections.resize(numVisible);

		for (int vi = 0; vi < numVisible; vi++)
			_hidden[hi]._visibleHiddenConnections[vi]._weight = weightDistVisibleHidden(generator);

		_hidden[hi]._hiddenHiddenConnections.resize(numHidden);

		for (int hio = 0; hio < numHidden; hio++)
			_hidden[hi]._hiddenHiddenConnections[hio]._weight = weightDistHiddenHidden(generator);
	}
}

void SparseCoder::activate(float activationLeak, float sparsity) {
	// Activate
	for (int hi = 0; hi < _hidden.size(); hi++) {
		float sum = 0.0f;// _hidden[hi]._bias._weight;

		for (int vi = 0; vi < _visible.size(); vi++)
			sum += _hidden[hi]._visibleHiddenConnections[vi]._weight * _visible[vi]._input;

		_hidden[hi]._activation = std::max(0.0f, sum);
	}

	// Inhibit
	for (int hi = 0; hi < _hidden.size(); hi++) {
		float sum = std::max(activationLeak, _hidden[hi]._activation);

		for (int hio = 0; hio < _hidden.size(); hio++)
			sum -= _hidden[hi]._hiddenHiddenConnections[hio]._weight * _hidden[hio]._activation;

		_hidden[hi]._state = std::max(0.0f, sum);
	}
}

void SparseCoder::reconstruct() {
	for (int vi = 0; vi < _visible.size(); vi++) {
		float sum = 0.0f;

		for (int hi = 0; hi < _hidden.size(); hi++)
			sum += _hidden[hi]._visibleHiddenConnections[vi]._weight * _hidden[hi]._state;

		_visible[vi]._reconstruction = sum;
	}
}

void SparseCoder::learn(float alpha, float beta, float gamma, float sparsity) {
	std::vector<float> visibleErrors(_visible.size());

	for (int vi = 0; vi < _visible.size(); vi++)
		visibleErrors[vi] = _visible[vi]._input - _visible[vi]._reconstruction;

	float sparsitySquared = sparsity * sparsity;

	for (int hi = 0; hi < _hidden.size(); hi++) {
		if (_hidden[hi]._state != 0.0f) {
			for (int vi = 0; vi < _visible.size(); vi++)
				_hidden[hi]._visibleHiddenConnections[vi]._weight += alpha * _hidden[hi]._state * (_visible[vi]._input - _hidden[hi]._state * _hidden[hi]._visibleHiddenConnections[vi]._weight);
		}

		for (int hio = 0; hio < _hidden.size(); hio++)
			_hidden[hi]._hiddenHiddenConnections[hio]._weight = std::max(0.0f, _hidden[hi]._hiddenHiddenConnections[hio]._weight + beta * (_hidden[hi]._state * _hidden[hio]._state - sparsitySquared));

		_hidden[hi]._hiddenHiddenConnections[hi]._weight = 0.0f;

		_hidden[hi]._bias._weight += gamma * -_hidden[hi]._activation;
	}
}

void SparseCoder::stepEnd() {
	for (int hi = 0; hi < _hidden.size(); hi++)
		_hidden[hi]._statePrev = _hidden[hi]._state;
}