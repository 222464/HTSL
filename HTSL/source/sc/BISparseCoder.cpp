#include "BISparseCoder.h"

#include <algorithm>

#include <assert.h>

#include <iostream>

using namespace sc;

void BISparseCoder::createRandom(int visibleSize, int hiddenSize, float weightScale, std::mt19937 &generator) {
	std::uniform_real_distribution<float> weightDist(0.0, 1.0);

	_visibleSize = visibleSize;
	_hiddenSize = hiddenSize;

	_visible.resize(_visibleSize);

	_hidden.resize(_hiddenSize);

	for (int hi = 0; hi < _hiddenSize; hi++) {
		// Receptive
		_hidden[hi]._visibleHiddenConnections.resize(_visible.size());

		float dist2 = 0.0f;

		for (int vi = 0; vi < _visible.size(); vi++) {
			VisibleConnection c;

			c._weight = weightDist(generator) * 2.0f - 1.0f;

			dist2 += c._weight * c._weight;

			_hidden[hi]._visibleHiddenConnections[vi] = c;
		}

		float normFactor = 1.0f / std::sqrt(dist2);

		for (int ci = 0; ci < _hidden[hi]._visibleHiddenConnections.size(); ci++)
			_hidden[hi]._visibleHiddenConnections[ci]._weight *= normFactor * weightScale;
	}
}

void BISparseCoder::activate(int iter, float stepSize, float lambda, float initActivationStdDev, std::mt19937 &generator) {
	std::normal_distribution<float> initActivationDist(0.0f, initActivationStdDev);

	for (int hi = 0; hi < _hidden.size(); hi++)
		_hidden[hi]._activation = initActivationDist(generator);

	reconstruct();

	for (int i = 0; i < iter; i++) {
		// Activate - deltaH = alpha * (D * (x - Dh) - lambda * h / (sqrt(h^2 + e)))
		for (int hi = 0; hi < _hidden.size(); hi++) {
			float sum = 0.0f;

			for (int ci = 0; ci < _hidden[hi]._visibleHiddenConnections.size(); ci++)
				sum += (_visible[ci]._input - _visible[ci]._reconstruction) * _hidden[hi]._visibleHiddenConnections[ci]._weight;

			//-lambda * _hidden[hi]._activation / std::sqrt(_hidden[hi]._activation * _hidden[hi]._activation + epsilon)
			_hidden[hi]._activation += stepSize * sum;

			float r = _hidden[hi]._activation - stepSize * lambda * (_hidden[hi]._activation > 0.0f ? 1.0f : -1.0f);

			if ((_hidden[hi]._activation > 0.0f) != (r > 0.0f))
				_hidden[hi]._activation = 0.0f;
			else
				_hidden[hi]._activation = r;
		}

		reconstruct();
	}

	// Binary-ize
	for (int hi = 0; hi < _hidden.size(); hi++)
		if (_hidden[hi]._activation != 0.0f)
			_hidden[hi]._activation = 1.0f;

	reconstruct();
}

void BISparseCoder::reconstruct() {
	for (int vi = 0; vi < _visible.size(); vi++)
		_visible[vi]._reconstruction = 0.0f;

	for (int hi = 0; hi < _hidden.size(); hi++) {
		//if (_hidden[hi]._activation != 0.0f)
		for (int ci = 0; ci < _hidden[hi]._visibleHiddenConnections.size(); ci++)
			_visible[ci]._reconstruction += _hidden[hi]._visibleHiddenConnections[ci]._weight * _hidden[hi]._activation;
	}
}

void BISparseCoder::reconstruct(const std::vector<float> &hiddenStates, std::vector<float> &recon) {
	recon.clear();
	recon.assign(_visible.size(), 0.0f);

	for (int hi = 0; hi < _hidden.size(); hi++) {
		//if (hiddenStates[hi] != 0.0f)
		for (int ci = 0; ci < _hidden[hi]._visibleHiddenConnections.size(); ci++)
			recon[ci] += _hidden[hi]._visibleHiddenConnections[ci]._weight * hiddenStates[hi];
	}
}

void BISparseCoder::learn(float alpha) {
	std::vector<float> visibleErrors(_visible.size(), 0.0f);

	for (int vi = 0; vi < _visible.size(); vi++)
		visibleErrors[vi] = _visible[vi]._input - _visible[vi]._reconstruction;

	for (int hi = 0; hi < _hidden.size(); hi++) {
		//if (_hidden[hi]._activation != 0.0f)
		for (int ci = 0; ci < _hidden[hi]._visibleHiddenConnections.size(); ci++)
			_hidden[hi]._visibleHiddenConnections[ci]._weight += alpha * _hidden[hi]._activation * visibleErrors[ci];
	}
}