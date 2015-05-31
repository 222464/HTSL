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

#include "BayesianOptimizer.h"

#include <algorithm>
#include <iostream>

using namespace hyp;

BayesianOptimizer::BayesianOptimizer()
	: _xi(0.001f), _annealingIterations(32), _annealingSamples(32), _annealingPertStdDev(0.3f), _annealingDecay(0.9f)
{}

void BayesianOptimizer::create(size_t numVariables, const std::vector<float> &minBounds, const std::vector<float> &maxBounds) {
	_sampleField.create(numVariables, 1);

	_minBounds = minBounds;
	_maxBounds = maxBounds;

	assert(numVariables == _minBounds.size() && _minBounds.size() == _maxBounds.size());
}

void BayesianOptimizer::generateNewVariables(std::mt19937 &generator) {
	_currentVariables.resize(_sampleField.getXSize());

	// Totally random values if below 2 samples
	if (_sampleField.getNumSamples() < 2) {
		for (size_t xi = 0; xi < _currentVariables.size(); xi++) {
			std::uniform_real_distribution<float> distBounds(_minBounds[xi], _maxBounds[xi]);

			_currentVariables[xi] = distBounds(generator);
		}
	}
	else {
		std::normal_distribution<float> distAnnealingPert(0.0f, _annealingPertStdDev);

		float stdDevMult = 1.0f;

		// Anneal a better sample point
		for (int i = 0; i < _annealingIterations; i++) {
			std::vector<float> aquisitionVariable = _currentVariables;

			float maxAquisition = -99999.0f;

			for (int s = 0; s < _annealingSamples; s++) {
				std::vector<float> sampleVariables(_currentVariables.size());

				for (size_t x = 0; x < _currentVariables.size(); x++)
					sampleVariables[x] = std::max(_minBounds[x], std::min(_maxBounds[x], _currentVariables[x] + stdDevMult * distAnnealingPert(generator)));

				float aquisition = (_sampleField.getYAtX(sampleVariables)[0] - _sampleField.getYAtX(_currentVariables)[0] - _xi) / std::max(0.00001f, std::sqrt(_sampleField.getVarianceAtX(sampleVariables)));
				
				if (aquisition > maxAquisition) {
					maxAquisition = aquisition;
					aquisitionVariable = sampleVariables;
				}
			}

			_currentVariables = aquisitionVariable;

			stdDevMult *= _annealingDecay;
		}
	}
}

void BayesianOptimizer::update(float fitness) {
	SampleField::Sample s;
	s._x = _currentVariables;
	s._y = std::vector<float>(1, fitness);

	_sampleField.addSample(s);
}

float hyp::normCDF(float x) {
	const float a1 = 0.254829592;
	const float a2 = -0.284496736;
	const float a3 = 1.421413741;
	const float a4 = -1.453152027;
	const float a5 = 1.061405429;
	const float p = 0.3275911;
	const float sqrt2Inv = 1.0f / std::sqrt(2.0f);

	int s = 1;

	if (x < 0)
		s = -1;

	x = std::abs(x) * sqrt2Inv;

	float t = 1.0f / (1.0f + p * x);
	float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * std::exp(-x * x);

	return 0.5f * (1.0f + s * y);
}