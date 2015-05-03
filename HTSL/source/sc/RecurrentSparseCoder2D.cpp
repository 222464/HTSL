#include "RecurrentSparseCoder2D.h"

#include <algorithm>

#include <assert.h>

#include <iostream>

using namespace sc;

void RecurrentSparseCoder2D::createRandom(int visibleWidth, int visibleHeight, int hiddenWidth, int hiddenHeight, int receptiveRadius, int inhibitionRadius, int recurrentRadius, std::mt19937 &generator) {
	std::uniform_real_distribution<float> weightDist(0.0f, 1.0f);

	_visibleWidth = visibleWidth;
	_visibleHeight = visibleHeight;
	_hiddenWidth = hiddenWidth;
	_hiddenHeight = hiddenHeight;

	_receptiveRadius = receptiveRadius;
	_inhibitionRadius = inhibitionRadius;
	_recurrentRadius = recurrentRadius;

	int numVisible = visibleWidth * visibleHeight;
	int numHidden = hiddenWidth * hiddenHeight;
	int receptiveSize = std::pow(receptiveRadius * 2 + 1, 2);
	int inhibitionSize = std::pow(inhibitionRadius * 2 + 1, 2);
	int recurrentSize = std::pow(recurrentRadius * 2 + 1, 2);

	_visible.resize(numVisible);

	_hidden.resize(numHidden);

	float hiddenToVisibleWidth = static_cast<float>(visibleWidth) / static_cast<float>(hiddenWidth);
	float hiddenToVisibleHeight = static_cast<float>(visibleHeight) / static_cast<float>(hiddenHeight);

	for (int hi = 0; hi < numHidden; hi++) {
		int hx = hi % hiddenWidth;
		int hy = hi / hiddenWidth;

		int centerX = std::round(hx * hiddenToVisibleWidth);
		int centerY = std::round(hy * hiddenToVisibleHeight);

		_hidden[hi]._bias = 0.0f;

		// Receptive
		_hidden[hi]._visibleHiddenConnections.reserve(receptiveSize);

		float dist2 = 0.0f;

		for (int dx = -receptiveRadius; dx <= receptiveRadius; dx++)
			for (int dy = -receptiveRadius; dy <= receptiveRadius; dy++) {
				int vx = centerX + dx;
				int vy = centerY + dy;

				if (vx >= 0 && vx < visibleWidth && vy >= 0 && vy < visibleHeight) {
					int vi = vx + vy * visibleWidth;

					VisibleConnection c;

					c._weight = weightDist(generator);
					c._index = vi;
					c._falloff = std::max(0.0f, 1.0f - std::sqrt(static_cast<float>(dx * dx + dy * dy)) / static_cast<float>(receptiveRadius + 1));

					dist2 += c._weight * c._weight;

					_hidden[hi]._visibleHiddenConnections.push_back(c);
				}
			}

		_hidden[hi]._visibleHiddenConnections.shrink_to_fit();

		float normFactor = 1.0f / dist2;

		for (int ci = 0; ci < _hidden[hi]._visibleHiddenConnections.size(); ci++)
			_hidden[hi]._visibleHiddenConnections[ci]._weight *= normFactor;

		// Inhibition
		_hidden[hi]._hiddenHiddenConnections.reserve(inhibitionSize);

		dist2 = 0.0f;

		for (int dx = -inhibitionRadius; dx <= inhibitionRadius; dx++)
			for (int dy = -inhibitionRadius; dy <= inhibitionRadius; dy++) {
				int hox = hx + dx;
				int hoy = hy + dy;

				if (hox >= 0 && hox < hiddenWidth && hoy >= 0 && hoy < hiddenWidth) {
					int hio = hox + hoy * hiddenWidth;

					HiddenConnection c;

					c._weight = -weightDist(generator);
					c._index = hio;
					c._falloff = std::max(0.0f, 1.0f - std::sqrt(static_cast<float>(dx * dx + dy * dy)) / static_cast<float>(inhibitionRadius + 1));

					dist2 += c._weight * c._weight;

					_hidden[hi]._hiddenHiddenConnections.push_back(c);
				}
			}

		normFactor = 1.0f / dist2;

		for (int ci = 0; ci < _hidden[hi]._hiddenHiddenConnections.size(); ci++)
			_hidden[hi]._hiddenHiddenConnections[ci]._weight *= normFactor;

		// Recurrent
		_hidden[hi]._hiddenPrevHiddenConnections.reserve(inhibitionSize);

		dist2 = 0.0f;

		for (int dx = -recurrentRadius; dx <= recurrentRadius; dx++)
			for (int dy = -recurrentRadius; dy <= recurrentRadius; dy++) {
				int hox = hx + dx;
				int hoy = hy + dy;

				if (hox >= 0 && hox < hiddenWidth && hoy >= 0 && hoy < hiddenWidth) {
					int hio = hox + hoy * hiddenWidth;

					VisibleConnection c;

					c._weight = weightDist(generator);
					c._index = hio;
					c._falloff = std::max(0.0f, 1.0f - std::sqrt(static_cast<float>(dx * dx + dy * dy)) / static_cast<float>(recurrentRadius + 1));

					dist2 += c._weight * c._weight;

					_hidden[hi]._hiddenPrevHiddenConnections.push_back(c);
				}
			}

		normFactor = 1.0f / dist2;

		for (int ci = 0; ci < _hidden[hi]._hiddenPrevHiddenConnections.size(); ci++)
			_hidden[hi]._hiddenPrevHiddenConnections[ci]._weight *= normFactor;
	}
}

void RecurrentSparseCoder2D::activate() {
	// Activate
	for (int hi = 0; hi < _hidden.size(); hi++) {
		float sum = -_hidden[hi]._bias;

		for (int ci = 0; ci < _hidden[hi]._visibleHiddenConnections.size(); ci++)
			sum += _hidden[hi]._visibleHiddenConnections[ci]._weight *  _hidden[hi]._visibleHiddenConnections[ci]._falloff * _visible[_hidden[hi]._visibleHiddenConnections[ci]._index]._input;

		for (int ci = 0; ci < _hidden[hi]._hiddenPrevHiddenConnections.size(); ci++)
			sum += _hidden[hi]._hiddenPrevHiddenConnections[ci]._weight *  _hidden[hi]._hiddenPrevHiddenConnections[ci]._falloff * _hidden[_hidden[hi]._hiddenPrevHiddenConnections[ci]._index]._statePrev;

		_hidden[hi]._activation = std::max(0.0f, sum);
	}

	// Inhibit
	for (int hi = 0; hi < _hidden.size(); hi++) {
		float sum = _hidden[hi]._activation;

		for (int ci = 0; ci < _hidden[hi]._hiddenHiddenConnections.size(); ci++)
			sum += _hidden[hi]._hiddenHiddenConnections[ci]._weight * _hidden[hi]._hiddenHiddenConnections[ci]._falloff * (_hidden[_hidden[hi]._hiddenHiddenConnections[ci]._index]._activation > _hidden[hi]._activation ? 1.0f : 0.0f);

		_hidden[hi]._state = sum > 0.0f ? 1.0f : 0.0f;// std::max(0.0f, sum);
	}
}

void RecurrentSparseCoder2D::reconstruct() {
	for (int vi = 0; vi < _visible.size(); vi++)
		_visible[vi]._reconstruction = 0.0f;

	for (int hi = 0; hi < _hidden.size(); hi++)
		_hidden[hi]._reconstruction = 0.0f;

	for (int hi = 0; hi < _hidden.size(); hi++) {
		for (int ci = 0; ci < _hidden[hi]._visibleHiddenConnections.size(); ci++)
			_visible[_hidden[hi]._visibleHiddenConnections[ci]._index]._reconstruction += _hidden[hi]._visibleHiddenConnections[ci]._weight * _hidden[hi]._state;

		for (int ci = 0; ci < _hidden[hi]._hiddenPrevHiddenConnections.size(); ci++)
			_hidden[_hidden[hi]._hiddenPrevHiddenConnections[ci]._index]._reconstruction += _hidden[hi]._hiddenPrevHiddenConnections[ci]._weight * _hidden[hi]._state;
	}
}

void RecurrentSparseCoder2D::learn(float alpha, float beta, float gamma, float delta, float sparsity) {
	std::vector<float> visibleErrors(_visible.size());
	std::vector<float> hiddenErrors(_hidden.size());

	for (int vi = 0; vi < _visible.size(); vi++)
		visibleErrors[vi] = _visible[vi]._input - _visible[vi]._reconstruction;

	for (int hi = 0; hi < _hidden.size(); hi++)
		hiddenErrors[hi] = _hidden[hi]._state - _hidden[hi]._reconstruction;

	float sparsitySquared = sparsity * sparsity;

	for (int hi = 0; hi < _hidden.size(); hi++) {
		float bit = _hidden[hi]._state > 0.0f ? 1.0f : 0.0f;
		float bitPrev = _hidden[hi]._statePrev > 0.0f ? 1.0f : 0.0f;

		for (int ci = 0; ci < _hidden[hi]._visibleHiddenConnections.size(); ci++)
			_hidden[hi]._visibleHiddenConnections[ci]._weight += beta * bit * (visibleErrors[_hidden[hi]._visibleHiddenConnections[ci]._index] + delta * _visible[_hidden[hi]._visibleHiddenConnections[ci]._index]._input);

		for (int ci = 0; ci < _hidden[hi]._hiddenPrevHiddenConnections.size(); ci++)
			_hidden[hi]._hiddenPrevHiddenConnections[ci]._weight += beta * bit * (hiddenErrors[_hidden[hi]._hiddenPrevHiddenConnections[ci]._index] + delta * _hidden[_hidden[hi]._hiddenPrevHiddenConnections[ci]._index]._statePrev);

		for (int ci = 0; ci < _hidden[hi]._hiddenHiddenConnections.size(); ci++) {
			if (_hidden[hi]._hiddenHiddenConnections[ci]._index == hi)
				_hidden[hi]._hiddenHiddenConnections[ci]._weight = 0.0f;
			else
				_hidden[hi]._hiddenHiddenConnections[ci]._weight = std::min(0.0f, _hidden[hi]._hiddenHiddenConnections[ci]._weight - alpha * ((_hidden[_hidden[hi]._hiddenHiddenConnections[ci]._index]._state > 0.0f ? 1.0f : 0.0f) * bit - sparsitySquared));
		}

		_hidden[hi]._bias += gamma * (bit - sparsity);
	}
}

void RecurrentSparseCoder2D::getVHWeights(int hx, int hy, std::vector<float> &rectangle) const {
	float hiddenToVisibleWidth = static_cast<float>(_visibleWidth) / static_cast<float>(_hiddenWidth);
	float hiddenToVisibleHeight = static_cast<float>(_visibleHeight) / static_cast<float>(_hiddenHeight);

	int dim = _receptiveRadius * 2 + 1;

	rectangle.resize(dim * dim, 0.0f);

	int hi = hx + hy * _hiddenWidth;

	int centerX = std::round(hx * hiddenToVisibleWidth);
	int centerY = std::round(hy * hiddenToVisibleHeight);

	for (int ci = 0; ci < _hidden[hi]._visibleHiddenConnections.size(); ci++) {
		int index = _hidden[hi]._visibleHiddenConnections[ci]._index;

		int vx = index % _visibleWidth;
		int vy = index / _visibleWidth;

		int dx = vx - centerX;
		int dy = vy - centerY;

		int rx = dx + _receptiveRadius;
		int ry = dy + _receptiveRadius;

		rectangle[rx + ry * dim] = _hidden[hi]._visibleHiddenConnections[ci]._weight;
	}
}

void RecurrentSparseCoder2D::stepEnd() {
	for (int hi = 0; hi < _hidden.size(); hi++)
		_hidden[hi]._statePrev = _hidden[hi]._state;
}