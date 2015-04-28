#include "SparseCoder2D.h"

#include <algorithm>

using namespace sc;

void SparseCoder2D::createRandom(int visibleWidth, int visibleHeight, int hiddenWidth, int hiddenHeight, int receptiveRadius, int inhibitionRadius, std::mt19937 &generator) {
	std::uniform_real_distribution<float> weightDist(0.0f, 1.0f);

	_visibleWidth = visibleWidth;
	_visibleHeight = visibleHeight;
	_hiddenWidth = hiddenWidth;
	_hiddenHeight = hiddenHeight;

	_receptiveRadius = receptiveRadius;
	_inhibitionRadius = inhibitionRadius;

	int numVisible = visibleWidth * visibleHeight;
	int numHidden = hiddenWidth * hiddenHeight;
	int receptiveSize = std::pow(receptiveRadius * 2 + 1, 2);
	int inhibitionSize = std::pow(inhibitionRadius * 2 + 1, 2);

	_visible.resize(numVisible);

	_hidden.resize(numHidden);

	float hiddenToVisibleWidth = static_cast<float>(visibleWidth) / static_cast<float>(hiddenWidth);
	float hiddenToVisibleHeight = static_cast<float>(visibleHeight) / static_cast<float>(hiddenHeight);

	for (int hi = 0; hi < numHidden; hi++) {
		int hx = hi % hiddenWidth;
		int hy = hi / hiddenWidth;

		int centerX = std::round(hx * hiddenToVisibleWidth);
		int centerY = std::round(hy * hiddenToVisibleWidth);

		_hidden[hi]._bias = 0.0f;

		_hidden[hi]._visibleHiddenConnections.reserve(receptiveSize);

		float dist2 = 0.0f;

		for (int dx = -receptiveRadius; dx <= receptiveRadius; dx++)
			for (int dy = -receptiveRadius; dy <= receptiveRadius; dy++) {
				int vx = centerX + dx;
				int vy = centerY + dy;

				if (vx >= 0 && vx < visibleWidth && vy >= 0 && vy < visibleHeight) {
					int vi = vx + vy * visibleWidth;

					Connection c;

					c._weight = weightDist(generator);
					c._index = vi;

					dist2 += c._weight * c._weight;

					_hidden[hi]._visibleHiddenConnections.push_back(c);
				}
			}

		_hidden[hi]._visibleHiddenConnections.shrink_to_fit();

		float normFactor = 1.0f / dist2;

		for (int ci = 0; ci < _hidden[hi]._visibleHiddenConnections.size(); ci++)
			_hidden[hi]._visibleHiddenConnections[ci]._weight *= normFactor;

		_hidden[hi]._hiddenHiddenConnections.reserve(inhibitionSize);

		dist2 = 0.0f;

		for (int dx = -inhibitionRadius; dx <= inhibitionRadius; dx++)
			for (int dy = -inhibitionRadius; dy <= inhibitionRadius; dy++) {
				int hox = hx + dx;
				int hoy = hy + dy;

				if (hox >= 0 && hox < hiddenWidth && hoy >= 0 && hoy < hiddenWidth) {
					int hio = hox + hoy * hiddenWidth;

					Connection c;

					c._weight = weightDist(generator);
					c._index = hio;

					dist2 += c._weight * c._weight;

					_hidden[hi]._hiddenHiddenConnections.push_back(c);
				}
			}

		normFactor = 1.0f / dist2;

		for (int ci = 0; ci < _hidden[hi]._hiddenHiddenConnections.size(); ci++)
			_hidden[hi]._hiddenHiddenConnections[ci]._weight *= normFactor;
	}
}

void SparseCoder2D::activate() {
	// Activate
	for (int hi = 0; hi < _hidden.size(); hi++) {
		float sum = -_hidden[hi]._bias;

		for (int ci = 0; ci < _hidden[hi]._visibleHiddenConnections.size(); ci++)
			sum += _hidden[hi]._visibleHiddenConnections[ci]._weight * _visible[_hidden[hi]._visibleHiddenConnections[ci]._index]._input;

		_hidden[hi]._activation = sum;
	}

	// Inhibit
	for (int hi = 0; hi < _hidden.size(); hi++) {
		float sum = _hidden[hi]._activation;

		for (int ci = 0; ci < _hidden[hi]._hiddenHiddenConnections.size(); ci++)
			sum += _hidden[hi]._hiddenHiddenConnections[ci]._weight * (_hidden[_hidden[hi]._hiddenHiddenConnections[ci]._index]._activation > _hidden[hi]._activation ? 1.0f : 0.0f);

		_hidden[hi]._state = std::max(0.0f, sum);
	}
}

void SparseCoder2D::reconstruct() {
	for (int vi = 0; vi < _visible.size(); vi++)
		_visible[vi]._reconstruction = 0.0f;

	for (int hi = 0; hi < _hidden.size(); hi++) {
		for (int ci = 0; ci < _hidden[hi]._visibleHiddenConnections.size(); ci++)
			_visible[_hidden[hi]._visibleHiddenConnections[ci]._index]._reconstruction += _hidden[hi]._visibleHiddenConnections[ci]._weight * _hidden[hi]._state;
	}
}

void SparseCoder2D::learn(float alpha, float beta, float gamma, float sparsity) {
	std::vector<float> visibleErrors(_visible.size());

	for (int vi = 0; vi < _visible.size(); vi++)
		visibleErrors[vi] = _visible[vi]._input - _visible[vi]._reconstruction;

	float sparsitySquared = sparsity * sparsity;

	for (int hi = 0; hi < _hidden.size(); hi++) {
		float bit = _hidden[hi]._state > 0.0f ? 1.0f : 0.0f;

		for (int ci = 0; ci < _hidden[hi]._visibleHiddenConnections.size(); ci++)
			_hidden[hi]._visibleHiddenConnections[ci]._weight += beta * bit * visibleErrors[_hidden[hi]._visibleHiddenConnections[ci]._index];// (_visible[vi]._input - _hidden[hi]._activation * _hidden[hi]._visibleHiddenConnections[vi]._weight);

		for (int ci = 0; ci < _hidden[hi]._hiddenHiddenConnections.size(); ci++) {
			if (_hidden[hi]._hiddenHiddenConnections[ci]._index == hi)
				_hidden[hi]._hiddenHiddenConnections[ci]._weight = 0.0f;
			else
				_hidden[hi]._hiddenHiddenConnections[ci]._weight = std::min(0.0f, _hidden[hi]._hiddenHiddenConnections[ci]._weight - alpha * (bit - sparsity) * (_hidden[_hidden[hi]._hiddenHiddenConnections[ci]._index]._activation > _hidden[hi]._activation ? 1.0f : 0.0f));
		}

		_hidden[hi]._bias += gamma * (bit - sparsity);
	}
}

void SparseCoder2D::getVHWeights(int hx, int hy, std::vector<float> &rectangle) const {
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