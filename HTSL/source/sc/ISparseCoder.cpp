#include "ISparseCoder.h"

#include <algorithm>

#include <assert.h>

#include <iostream>

using namespace sc;

void ISparseCoder::createRandom(int visibleSize, int hiddenSize, double weightScale, std::mt19937 &generator) {
	std::uniform_real_distribution<double> weightDist(0.0, 1.0);

	_visibleSize = visibleSize;
	_hiddenSize = hiddenSize;

	_visible.resize(_visibleSize);

	_hidden.resize(_hiddenSize);

	for (int hi = 0; hi < _hiddenSize; hi++) {
		// Receptive
		_hidden[hi]._visibleHiddenConnections.resize(_visible.size());

		double dist2 = 0.0;

		for (int vi = 0; vi < _visible.size(); vi++) {
			VisibleConnection c;

			c._weight = weightDist(generator) * 2.0 - 1.0;

			dist2 += c._weight * c._weight;

			_hidden[hi]._visibleHiddenConnections[vi] = c;
		}

		double normFactor = 1.0 / std::sqrt(dist2);

		for (int ci = 0; ci < _hidden[hi]._visibleHiddenConnections.size(); ci++)
			_hidden[hi]._visibleHiddenConnections[ci]._weight *= normFactor * weightScale;
	}
}

void ISparseCoder::activate(int iter, double stepSize, double lambda, double epsilon) {
	for (int i = 0; i < iter; i++) {
		// Activate - deltaH = alpha * (D * (x - Dh) - lambda * h / (sqrt(h^2 + e)))
		for (int hi = 0; hi < _hidden.size(); hi++) {
			double sum = 0.0;

			for (int ci = 0; ci < _hidden[hi]._visibleHiddenConnections.size(); ci++) {
				sum += (_visible[ci]._input - _visible[ci]._reconstruction) * _hidden[hi]._visibleHiddenConnections[ci]._weight;
			}

			_hidden[hi]._activation += stepSize * (sum - lambda * _hidden[hi]._activation / std::sqrt(_hidden[hi]._activation * _hidden[hi]._activation + epsilon));
		}

		reconstruct();
	}
}

void ISparseCoder::reconstruct() {
	for (int vi = 0; vi < _visible.size(); vi++)
		_visible[vi]._reconstruction = 0.0;

	for (int hi = 0; hi < _hidden.size(); hi++) {
		for (int ci = 0; ci < _hidden[hi]._visibleHiddenConnections.size(); ci++)
			_visible[ci]._reconstruction += _hidden[hi]._visibleHiddenConnections[ci]._weight * _hidden[hi]._activation;
	}
}

void ISparseCoder::reconstruct(const std::vector<double> &hiddenStates, std::vector<double> &recon) {
	recon.clear();
	recon.assign(_visible.size(), 0.0);

	for (int hi = 0; hi < _hidden.size(); hi++) {
		for (int ci = 0; ci < _hidden[hi]._visibleHiddenConnections.size(); ci++)
			recon[ci] += _hidden[hi]._visibleHiddenConnections[ci]._weight * hiddenStates[hi];
	}
}

void ISparseCoder::learn(double alpha) {
	std::vector<double> visibleErrors(_visible.size(), 0.0);

	for (int vi = 0; vi < _visible.size(); vi++)
		visibleErrors[vi] = _visible[vi]._input - _visible[vi]._reconstruction;

	for (int hi = 0; hi < _hidden.size(); hi++) {
		double learn = _hidden[hi]._activation;

		for (int ci = 0; ci < _hidden[hi]._visibleHiddenConnections.size(); ci++)
			_hidden[hi]._visibleHiddenConnections[ci]._weight += alpha * learn * visibleErrors[ci];
	}
}