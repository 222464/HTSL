#include "SparseCoder.h"

#include <algorithm>

#include <assert.h>

#include <iostream>

using namespace sc;

void SparseCoder::createRandom(int visibleSize, int hiddenSize, int receptiveRadius, int inhibitionRadius, double weightScale, std::mt19937 &generator) {
	std::uniform_real_distribution<double> weightDist(0.0, 1.0);

	_visibleSize = visibleSize;
	_hiddenSize = hiddenSize;

	_receptiveRadius = receptiveRadius;
	_inhibitionRadius = inhibitionRadius;

	int receptiveSize = std::pow(receptiveRadius * 2 + 1, 2);
	int inhibitionSize = std::pow(inhibitionRadius * 2 + 1, 2);

	_visible.resize(_visibleSize);

	_hidden.resize(_hiddenSize);

	double hiddenToVisible = static_cast<double>(_visibleSize - 1) / static_cast<double>(_hiddenSize - 1);

	for (int hi = 0; hi < _hiddenSize; hi++) {
		int center = std::round(hi * hiddenToVisible);

		_hidden[hi]._bias = 0.0;

		// Receptive
		_hidden[hi]._visibleHiddenConnections.reserve(receptiveSize);

		double dist2 = 0.0;

		for (int d = -receptiveRadius; d <= receptiveRadius; d++) {
			int v = center + d;

			if (v >= 0 && v < _visibleSize) {
				VisibleConnection c;

				c._weight = weightDist(generator) * 2.0 - 1.0;
				c._index = v;
		
				dist2 += c._weight * c._weight;

				_hidden[hi]._visibleHiddenConnections.push_back(c);
			}
		}

		_hidden[hi]._visibleHiddenConnections.shrink_to_fit();

		double normFactor = 1.0 / std::sqrt(dist2);

		for (int ci = 0; ci < _hidden[hi]._visibleHiddenConnections.size(); ci++)
			_hidden[hi]._visibleHiddenConnections[ci]._weight *= normFactor * weightScale;

		// Inhibition
		_hidden[hi]._hiddenHiddenConnections.reserve(inhibitionSize);

		dist2 = 0.0;

		for (int d = -inhibitionRadius; d <= inhibitionRadius; d++) {
			if (d == 0)
				continue;

			int h = hi + d;

			if (h >= 0 && h < _hiddenSize) {
				HiddenConnection c;

				c._weight = weightDist(generator);
				c._index = h;
	
				dist2 += c._weight * c._weight;

				_hidden[hi]._hiddenHiddenConnections.push_back(c);
			}
		}

		_hidden[hi]._hiddenHiddenConnections.shrink_to_fit();

		normFactor = 1.0 / std::sqrt(dist2);

		for (int ci = 0; ci < _hidden[hi]._hiddenHiddenConnections.size(); ci++)
			_hidden[hi]._hiddenHiddenConnections[ci]._weight *= normFactor * weightScale;
	}
}

void SparseCoder::activate() {
	// Activate
	for (int hi = 0; hi < _hidden.size(); hi++) {
		double sum = 0.0;

		for (int ci = 0; ci < _hidden[hi]._visibleHiddenConnections.size(); ci++) {
			double delta = _visible[_hidden[hi]._visibleHiddenConnections[ci]._index]._input - _hidden[hi]._visibleHiddenConnections[ci]._weight;

			sum += -delta * delta;
		}

		_hidden[hi]._activation = sum;
	}

	// Inhibit
	for (int hi = 0; hi < _hidden.size(); hi++) {
		double inhibition = _hidden[hi]._bias;

		for (int ci = 0; ci < _hidden[hi]._hiddenHiddenConnections.size(); ci++)
			inhibition += _hidden[hi]._hiddenHiddenConnections[ci]._weight * (_hidden[_hidden[hi]._hiddenHiddenConnections[ci]._index]._activation > _hidden[hi]._activation ? 1.0 : 0.0);

		_hidden[hi]._state = (1.0 - inhibition) > 0.0 ? 1.0 : 0.0;
	}
}

void SparseCoder::reconstruct() {
	std::vector<double> visibleSums(_visible.size(), 0.0);

	for (int vi = 0; vi < _visible.size(); vi++)
		_visible[vi]._reconstruction = 0.0;

	for (int hi = 0; hi < _hidden.size(); hi++) {
		for (int ci = 0; ci < _hidden[hi]._visibleHiddenConnections.size(); ci++) {
			_visible[_hidden[hi]._visibleHiddenConnections[ci]._index]._reconstruction += _hidden[hi]._visibleHiddenConnections[ci]._weight * _hidden[hi]._state;
			visibleSums[_hidden[hi]._visibleHiddenConnections[ci]._index] += _hidden[hi]._state;
		}
	}

	for (int vi = 0; vi < _visible.size(); vi++)
		_visible[vi]._reconstruction /= std::max(0.0001, visibleSums[vi]);
}

void SparseCoder::reconstruct(const std::vector<double> &hiddenStates, std::vector<double> &recon) {
	recon.clear();
	recon.assign(_visible.size(), 0.0);

	std::vector<double> visibleSums(_visible.size(), 0.0);

	for (int hi = 0; hi < _hidden.size(); hi++) {
		for (int ci = 0; ci < _hidden[hi]._visibleHiddenConnections.size(); ci++) {
			recon[_hidden[hi]._visibleHiddenConnections[ci]._index] += _hidden[hi]._visibleHiddenConnections[ci]._weight * hiddenStates[hi];
			visibleSums[_hidden[hi]._visibleHiddenConnections[ci]._index] += hiddenStates[hi];
		}
	}

	for (int vi = 0; vi < _visible.size(); vi++)
		recon[vi] /= std::max(0.0001, visibleSums[vi]);
}

void SparseCoder::learn(double alpha, double beta, double gamma, double sparsity) {
	std::vector<double> visibleErrors(_visible.size(), 0.0);

	for (int vi = 0; vi < _visible.size(); vi++)
		visibleErrors[vi] = _visible[vi]._input - _visible[vi]._reconstruction;

	double sparsitySquared = sparsity * sparsity;

	for (int hi = 0; hi < _hidden.size(); hi++) {
		double learn = _hidden[hi]._state;

		for (int ci = 0; ci < _hidden[hi]._visibleHiddenConnections.size(); ci++)
			_hidden[hi]._visibleHiddenConnections[ci]._weight += beta * learn * visibleErrors[_hidden[hi]._visibleHiddenConnections[ci]._index];

		for (int ci = 0; ci < _hidden[hi]._hiddenHiddenConnections.size(); ci++)
			_hidden[hi]._hiddenHiddenConnections[ci]._weight = std::max(0.0, _hidden[hi]._hiddenHiddenConnections[ci]._weight + alpha * (_hidden[hi]._state * (_hidden[_hidden[hi]._hiddenHiddenConnections[ci]._index]._activation < _hidden[hi]._activation ? 1.0 : 0.0) - sparsitySquared)); //_hidden[_hidden[hi]._hiddenHiddenConnections[ci]._index]._state * 

		_hidden[hi]._bias += gamma * (_hidden[hi]._state - sparsity);
	}
}