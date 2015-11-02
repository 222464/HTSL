#include "IRSDR.h"

#include <algorithm>

#include <assert.h>

#include <iostream>

using namespace sdr;

void IRSDR::createRandom(int visibleWidth, int visibleHeight, int hiddenWidth, int hiddenHeight, int receptiveRadius, int recurrentRadius, float initMinWeight, float initMaxWeight, std::mt19937 &generator) {
	std::uniform_real_distribution<float> weightDist(initMinWeight, initMaxWeight);

	_visibleWidth = visibleWidth;
	_visibleHeight = visibleHeight;
	_hiddenWidth = hiddenWidth;
	_hiddenHeight = hiddenHeight;

	_receptiveRadius = receptiveRadius;
	_recurrentRadius = recurrentRadius;

	int numVisible = visibleWidth * visibleHeight;
	int numHidden = hiddenWidth * hiddenHeight;
	int receptiveSize = std::pow(receptiveRadius * 2 + 1, 2);
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

		// Receptive
		_hidden[hi]._feedForwardConnections.reserve(receptiveSize);

		for (int dx = -receptiveRadius; dx <= receptiveRadius; dx++)
			for (int dy = -receptiveRadius; dy <= receptiveRadius; dy++) {
				int vx = centerX + dx;
				int vy = centerY + dy;

				if (vx >= 0 && vx < visibleWidth && vy >= 0 && vy < visibleHeight) {
					int vi = vx + vy * visibleWidth;

					Connection c;

					c._weight = weightDist(generator);
					c._index = vi;

					_hidden[hi]._feedForwardConnections.push_back(c);
				}
			}

		_hidden[hi]._feedForwardConnections.shrink_to_fit();

		// Recurrent
		if (recurrentRadius != -1) {
			_hidden[hi]._recurrentConnections.reserve(recurrentSize);

			for (int dx = -recurrentRadius; dx <= recurrentRadius; dx++)
				for (int dy = -recurrentRadius; dy <= recurrentRadius; dy++) {
					if (dx == 0 && dy == 0)
						continue;

					int hox = hx + dx;
					int hoy = hy + dy;

					if (hox >= 0 && hox < hiddenWidth && hoy >= 0 && hoy < hiddenHeight) {
						int hio = hox + hoy * hiddenWidth;

						Connection c;

						c._weight = weightDist(generator);
						c._index = hio;

						_hidden[hi]._recurrentConnections.push_back(c);
					}
				}

			_hidden[hi]._recurrentConnections.shrink_to_fit();
		}
	}
}

void IRSDR::activate(int iter, float stepSize, float lambda, float epsilon, float hiddenDecay, float initActivationNoise, std::mt19937 &generator) {
	std::uniform_real_distribution<float> initActivationDist(-initActivationNoise, initActivationNoise);
	
	for (int hi = 0; hi < _hidden.size(); hi++) {
		float sum = 0.0f;

		for (int ci = 0; ci < _hidden[hi]._feedForwardConnections.size(); ci++)
			sum += _visible[_hidden[hi]._feedForwardConnections[ci]._index]._input * _hidden[hi]._feedForwardConnections[ci]._weight;

		for (int ci = 0; ci < _hidden[hi]._recurrentConnections.size(); ci++)
			sum += _hidden[_hidden[hi]._recurrentConnections[ci]._index]._statePrev * _hidden[hi]._recurrentConnections[ci]._weight;

		_hidden[hi]._state = sum + initActivationDist(generator);
	}

	reconstruct();

	for (int i = 0; i < iter; i++) {
		std::vector<float> visibleErrors(_visible.size(), 0.0f);
		std::vector<float> hiddenErrors(_hidden.size(), 0.0f);

		for (int vi = 0; vi < _visible.size(); vi++)
			visibleErrors[vi] = _visible[vi]._input - _visible[vi]._reconstruction;

		for (int hi = 0; hi < _hidden.size(); hi++)
			hiddenErrors[hi] = _hidden[hi]._statePrev - _hidden[hi]._reconstruction;

		// Activate - deltaH = alpha * (D * (x - Dh) - lambda * h / (sqrt(h^2 + e)))
		for (int hi = 0; hi < _hidden.size(); hi++) {
			float sum = 0.0f;

			for (int ci = 0; ci < _hidden[hi]._feedForwardConnections.size(); ci++)
				sum += visibleErrors[_hidden[hi]._feedForwardConnections[ci]._index] * _hidden[hi]._feedForwardConnections[ci]._weight;

			for (int ci = 0; ci < _hidden[hi]._recurrentConnections.size(); ci++)
				sum += hiddenErrors[_hidden[hi]._recurrentConnections[ci]._index] * _hidden[hi]._recurrentConnections[ci]._weight;

			//-lambda * _hidden[hi]._state / std::sqrt(_hidden[hi]._state * _hidden[hi]._state + epsilon)
			//_hidden[hi]._state += stepSize * (sum - lambda * _hidden[hi]._state / std::sqrt(_hidden[hi]._state * _hidden[hi]._state + epsilon)) - hiddenDecay * _hidden[hi]._state;
		
			_hidden[hi]._state += stepSize * sum;
		
			float r = _hidden[hi]._state - stepSize * lambda * (_hidden[hi]._state > 0.0f ? 1.0f : -1.0f);

			if ((_hidden[hi]._state > 0.0f) != (r > 0.0f))
				_hidden[hi]._state = 0.0f;
			else
				_hidden[hi]._state = r;
		}

		reconstruct();

	}
}

void IRSDR::inhibit(int iter, float stepSize, float lambda, float epsilon, const std::vector<float> &activations, std::vector<float> &states, float initActivationStdDev, std::mt19937 &generator) {
	std::normal_distribution<float> initActivationDist(0.0f, initActivationStdDev);

	std::vector<float> reconHidden;
	std::vector<float> reconVisible;

	reconstruct(activations, reconHidden, reconVisible);

	std::vector<float> inputsHidden = reconHidden;
	std::vector<float> inputsVisible = reconVisible;
	
	if (states.size() != _hidden.size())
		states.resize(_hidden.size());

	for (int hi = 0; hi < _hidden.size(); hi++)
		states[hi] = initActivationDist(generator);

	for (int i = 0; i < iter; i++) {
		// Activate - deltaH = alpha * (D * (x - Dh) - lambda * h / (sqrt(h^2 + e)))
		for (int hi = 0; hi < _hidden.size(); hi++) {
			float sum = 0.0f;

			for (int ci = 0; ci < _hidden[hi]._feedForwardConnections.size(); ci++)
				sum += (inputsVisible[_hidden[hi]._feedForwardConnections[ci]._index] - reconVisible[_hidden[hi]._feedForwardConnections[ci]._index]) * _hidden[hi]._feedForwardConnections[ci]._weight;

			for (int ci = 0; ci < _hidden[hi]._recurrentConnections.size(); ci++)
				sum += (inputsHidden[_hidden[hi]._recurrentConnections[ci]._index] - reconHidden[_hidden[hi]._recurrentConnections[ci]._index]) * _hidden[hi]._recurrentConnections[ci]._weight;

			//-lambda * _hidden[hi]._state / std::sqrt(_hidden[hi]._state * _hidden[hi]._state + epsilon)
			states[hi] += stepSize * sum;

			float r = states[hi] - stepSize * lambda * (states[hi] > 0.0f ? 1.0f : -1.0f);

			if ((states[hi] > 0.0f) != (r > 0.0f))
				states[hi] = 0.0f;
			else
				states[hi] = r;
		}

		reconstruct(activations, reconHidden, reconVisible);
	}
}

void IRSDR::reconstruct() {
	std::vector<float> visibleDivs(_visible.size(), 0.0f);
	std::vector<float> hiddenDivs(_hidden.size(), 0.0f);

	for (int vi = 0; vi < _visible.size(); vi++)
		_visible[vi]._reconstruction = 0.0f;

	for (int hi = 0; hi < _hidden.size(); hi++)
		_hidden[hi]._reconstruction = 0.0f;

	for (int hi = 0; hi < _hidden.size(); hi++) {
		for (int ci = 0; ci < _hidden[hi]._feedForwardConnections.size(); ci++) {
			_visible[_hidden[hi]._feedForwardConnections[ci]._index]._reconstruction += _hidden[hi]._feedForwardConnections[ci]._weight * _hidden[hi]._state;

			visibleDivs[_hidden[hi]._feedForwardConnections[ci]._index] += _hidden[hi]._state;
		}

		for (int ci = 0; ci < _hidden[hi]._recurrentConnections.size(); ci++) {
			_hidden[_hidden[hi]._recurrentConnections[ci]._index]._reconstruction += _hidden[hi]._recurrentConnections[ci]._weight * _hidden[hi]._state;

			hiddenDivs[_hidden[hi]._recurrentConnections[ci]._index] += _hidden[hi]._state;
		}
	}

	//for (int vi = 0; vi < _visible.size(); vi++)
	//	_visible[vi]._reconstruction /= std::max(1.0f, visibleDivs[vi]);

	//for (int hi = 0; hi < _hidden.size(); hi++)
	//	_hidden[hi]._reconstruction /= std::max(1.0f, hiddenDivs[hi]);
}

void IRSDR::reconstruct(const std::vector<float> &states, std::vector<float> &reconHidden, std::vector<float> &reconVisible) {
	std::vector<float> visibleDivs(_visible.size(), 0.0f);
	std::vector<float> hiddenDivs(_hidden.size(), 0.0f);

	reconVisible.clear();
	reconVisible.assign(_visible.size(), 0.0f);

	reconHidden.clear();
	reconHidden.assign(_hidden.size(), 0.0f);

	for (int hi = 0; hi < _hidden.size(); hi++) {
		for (int ci = 0; ci < _hidden[hi]._feedForwardConnections.size(); ci++) {
			reconVisible[_hidden[hi]._feedForwardConnections[ci]._index] += _hidden[hi]._feedForwardConnections[ci]._weight * states[hi];

			visibleDivs[_hidden[hi]._feedForwardConnections[ci]._index] += states[hi];
		}

		for (int ci = 0; ci < _hidden[hi]._recurrentConnections.size(); ci++) {
			reconHidden[_hidden[hi]._recurrentConnections[ci]._index] += _hidden[hi]._recurrentConnections[ci]._weight * states[hi];

			hiddenDivs[_hidden[hi]._recurrentConnections[ci]._index] += states[hi];
		}
	}
}

void IRSDR::reconstructFeedForward(const std::vector<float> &states, std::vector<float> &recon) {
	std::vector<float> visibleDivs(_visible.size(), 0.0f);

	recon.clear();
	recon.assign(_visible.size(), 0.0f);

	for (int hi = 0; hi < _hidden.size(); hi++) {
		for (int ci = 0; ci < _hidden[hi]._feedForwardConnections.size(); ci++) {
			recon[_hidden[hi]._feedForwardConnections[ci]._index] += _hidden[hi]._feedForwardConnections[ci]._weight * states[hi];

			visibleDivs[_hidden[hi]._feedForwardConnections[ci]._index] += states[hi];
		}
	}

	//for (int vi = 0; vi < _visible.size(); vi++)
	//	recon[vi] /= std::max(1.0f, visibleDivs[vi]);
}

void IRSDR::learn(float learnFeedForward, float learnRecurrent, float gamma) {
	std::vector<float> visibleErrors(_visible.size(), 0.0f);
	std::vector<float> hiddenErrors(_hidden.size(), 0.0f);

	for (int vi = 0; vi < _visible.size(); vi++)
		visibleErrors[vi] = _visible[vi]._input - _visible[vi]._reconstruction;

	for (int hi = 0; hi < _hidden.size(); hi++)
		hiddenErrors[hi] = _hidden[hi]._statePrev - _hidden[hi]._reconstruction;

	for (int hi = 0; hi < _hidden.size(); hi++) {
		float learn = _hidden[hi]._state;// > 0.0f ? 1.0f : 0.0f;

		//if (_hidden[hi]._activation != 0.0f)
		for (int ci = 0; ci < _hidden[hi]._feedForwardConnections.size(); ci++)
			_hidden[hi]._feedForwardConnections[ci]._weight += learnFeedForward * learn * visibleErrors[_hidden[hi]._feedForwardConnections[ci]._index] - gamma * _hidden[hi]._feedForwardConnections[ci]._weight;

		for (int ci = 0; ci < _hidden[hi]._recurrentConnections.size(); ci++)
			_hidden[hi]._recurrentConnections[ci]._weight += learnRecurrent * learn * hiddenErrors[_hidden[hi]._recurrentConnections[ci]._index] - gamma * _hidden[hi]._recurrentConnections[ci]._weight;
	}
}

/*void IRSDR::learn(const std::vector<float> &attentions, float learnFeedForward, float learnRecurrent) {
	std::vector<float> visibleErrors(_visible.size(), 0.0f);
	std::vector<float> hiddenErrors(_hidden.size(), 0.0f);

	float error = 0.0f;

	for (int vi = 0; vi < _visible.size(); vi++)
		error += std::pow(_visible[vi]._input - _visible[vi]._reconstruction, 2);

	for (int hi = 0; hi < _hidden.size(); hi++)
		error += std::pow(_hidden[hi]._statePrev - _hidden[hi]._reconstruction, 2);

	std::cout << error << std::endl;

	for (int vi = 0; vi < _visible.size(); vi++)
		visibleErrors[vi] = _visible[vi]._input - _visible[vi]._reconstruction;

	for (int hi = 0; hi < _hidden.size(); hi++)
		hiddenErrors[hi] = _hidden[hi]._statePrev - _hidden[hi]._reconstruction;

	for (int hi = 0; hi < _hidden.size(); hi++) {
		//if (_hidden[hi]._activation != 0.0f)
		for (int ci = 0; ci < _hidden[hi]._feedForwardConnections.size(); ci++)
			_hidden[hi]._feedForwardConnections[ci]._weight += learnFeedForward * _hidden[hi]._state * attentions[hi] * visibleErrors[_hidden[hi]._feedForwardConnections[ci]._index];

		for (int ci = 0; ci < _hidden[hi]._recurrentConnections.size(); ci++)
			_hidden[hi]._recurrentConnections[ci]._weight += learnRecurrent * _hidden[hi]._state * attentions[hi] * hiddenErrors[_hidden[hi]._recurrentConnections[ci]._index];
	}
}*/

void IRSDR::getVHWeights(int hx, int hy, std::vector<float> &rectangle) const {
	float hiddenToVisibleWidth = static_cast<float>(_visibleWidth) / static_cast<float>(_hiddenWidth);
	float hiddenToVisibleHeight = static_cast<float>(_visibleHeight) / static_cast<float>(_hiddenHeight);

	int dim = _receptiveRadius * 2 + 1;

	rectangle.resize(dim * dim, 0.0f);

	int hi = hx + hy * _hiddenWidth;

	int centerX = std::round(hx * hiddenToVisibleWidth);
	int centerY = std::round(hy * hiddenToVisibleHeight);

	for (int ci = 0; ci < _hidden[hi]._feedForwardConnections.size(); ci++) {
		int index = _hidden[hi]._feedForwardConnections[ci]._index;

		int vx = index % _visibleWidth;
		int vy = index / _visibleWidth;

		int dx = vx - centerX;
		int dy = vy - centerY;

		int rx = dx + _receptiveRadius;
		int ry = dy + _receptiveRadius;

		rectangle[rx + ry * dim] = _hidden[hi]._feedForwardConnections[ci]._weight;
	}
}

void IRSDR::stepEnd() {
	for (int hi = 0; hi < _hidden.size(); hi++)
		_hidden[hi]._statePrev = _hidden[hi]._state;
}