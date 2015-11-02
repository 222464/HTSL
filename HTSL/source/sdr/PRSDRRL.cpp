#include "PRSDRRL.h"

#include <SFML/Window.hpp>
#include <iostream>

#include <algorithm>

using namespace sdr;

void PRSDRRL::createRandom(int inputWidth, int inputHeight, int inputFeedBackRadius, const std::vector<InputType> &inputTypes, const std::vector<LayerDesc> &layerDescs, float initMinWeight, float initMaxWeight, float initThreshold, std::mt19937 &generator) {
	std::uniform_real_distribution<float> weightDist(initMinWeight, initMaxWeight);

	_inputTypes = inputTypes;

	for (int i = 0; i < _inputTypes.size(); i++) {
		if (_inputTypes[i] == _q)
			_qInputIndices.push_back(i);
		else if (_inputTypes[i] == _action)
			_actionInputIndices.push_back(i);
	}

	_qInputOffsets.resize(_qInputIndices.size());

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	for (int i = 0; i < _qInputOffsets.size(); i++)
		_qInputOffsets[i] = dist01(generator);

	_layerDescs = layerDescs;

	_layers.resize(_layerDescs.size());

	int widthPrev = inputWidth;
	int heightPrev = inputHeight;

	for (int l = 0; l < _layerDescs.size(); l++) {
		//_layers[l]._sdr.createRandom(widthPrev, heightPrev, _layerDescs[l]._width, _layerDescs[l]._height, _layerDescs[l]._receptiveRadius, _layerDescs[l]._lateralRadius, _layerDescs[l]._recurrentRadius, initMinWeight, initMaxWeight, initThreshold, generator);

		_layers[l]._predictionNodes.resize(_layerDescs[l]._width * _layerDescs[l]._height);

		int feedBackSize = std::pow(_layerDescs[l]._feedBackRadius * 2 + 1, 2);
		int predictiveSize = std::pow(_layerDescs[l]._predictiveRadius * 2 + 1, 2);

		float hiddenToNextHiddenWidth = 1.0f;
		float hiddenToNextHiddenHeight = 1.0f;

		if (l < _layers.size() - 1) {
			hiddenToNextHiddenWidth = static_cast<float>(_layerDescs[l + 1]._width) / static_cast<float>(_layerDescs[l]._width);
			hiddenToNextHiddenHeight = static_cast<float>(_layerDescs[l + 1]._height) / static_cast<float>(_layerDescs[l]._height);
		}

		for (int pi = 0; pi < _layers[l]._predictionNodes.size(); pi++) {
			PredictionNode &p = _layers[l]._predictionNodes[pi];

			p._bias._weight = weightDist(generator);

			int hx = pi % _layerDescs[l]._width;
			int hy = pi / _layerDescs[l]._width;

			// Feed Back
			if (l < _layers.size() - 1) {
				p._feedBackConnections.reserve(feedBackSize);

				int centerX = std::round(hx * hiddenToNextHiddenWidth);
				int centerY = std::round(hy * hiddenToNextHiddenHeight);

				for (int dx = -_layerDescs[l]._feedBackRadius; dx <= _layerDescs[l]._feedBackRadius; dx++)
					for (int dy = -_layerDescs[l]._feedBackRadius; dy <= _layerDescs[l]._feedBackRadius; dy++) {
						int hox = centerX + dx;
						int hoy = centerY + dy;

						if (hox >= 0 && hox < _layerDescs[l + 1]._width && hoy >= 0 && hoy < _layerDescs[l + 1]._height) {
							int hio = hox + hoy * _layerDescs[l + 1]._width;

							Connection c;

							c._weight = weightDist(generator);
							c._index = hio;

							p._feedBackConnections.push_back(c);
						}
					}

				p._feedBackConnections.shrink_to_fit();
			}

			// Predictive
			p._predictiveConnections.reserve(feedBackSize);

			for (int dx = -_layerDescs[l]._predictiveRadius; dx <= _layerDescs[l]._predictiveRadius; dx++)
				for (int dy = -_layerDescs[l]._predictiveRadius; dy <= _layerDescs[l]._predictiveRadius; dy++) {
					int hox = hx + dx;
					int hoy = hy + dy;

					if (hox >= 0 && hox < _layerDescs[l]._width && hoy >= 0 && hoy < _layerDescs[l]._height) {
						int hio = hox + hoy * _layerDescs[l]._width;

						Connection c;

						c._weight = weightDist(generator);
						c._index = hio;

						p._predictiveConnections.push_back(c);
					}
				}

			p._predictiveConnections.shrink_to_fit();
		}

		widthPrev = _layerDescs[l]._width;
		heightPrev = _layerDescs[l]._height;
	}

	_inputPredictionNodes.resize(_inputTypes.size());

	float inputToNextHiddenWidth = static_cast<float>(_layerDescs.front()._width) / static_cast<float>(inputWidth);
	float inputToNextHiddenHeight = static_cast<float>(_layerDescs.front()._height) / static_cast<float>(inputHeight);

	for (int pi = 0; pi < _inputPredictionNodes.size(); pi++) {
		PredictionNode &p = _inputPredictionNodes[pi];

		p._bias._weight = weightDist(generator);

		int hx = pi % inputWidth;
		int hy = pi / inputWidth;

		int feedBackSize = std::pow(inputFeedBackRadius * 2 + 1, 2);

		// Feed Back
		p._feedBackConnections.reserve(feedBackSize);

		int centerX = std::round(hx * inputToNextHiddenWidth);
		int centerY = std::round(hy * inputToNextHiddenHeight);

		for (int dx = -inputFeedBackRadius; dx <= inputFeedBackRadius; dx++)
			for (int dy = -inputFeedBackRadius; dy <= inputFeedBackRadius; dy++) {
				int hox = centerX + dx;
				int hoy = centerY + dy;

				if (hox >= 0 && hox < _layerDescs.front()._width && hoy >= 0 && hoy < _layerDescs.front()._height) {
					int hio = hox + hoy * _layerDescs.front()._width;

					Connection c;

					c._weight = weightDist(generator);
					c._index = hio;

					p._feedBackConnections.push_back(c);
				}
			}

		p._feedBackConnections.shrink_to_fit();
	}
}

void PRSDRRL::simStep(float reward, std::mt19937 &generator, bool learn) {
	// Feature extraction
	for (int l = 0; l < _layers.size(); l++) {
		//_layers[l]._sdr.activate(_layerDescs[l]._sparsity);

		//_layers[l]._sdr.reconstruct();

		// Set inputs for next layer if there is one
		if (l < _layers.size() - 1) {
			for (int i = 0; i < _layers[l]._sdr.getNumHidden(); i++)
				_layers[l + 1]._sdr.setVisibleState(i, _layers[l]._sdr.getHiddenState(i));
		}
	}

	// Prediction
	std::vector<std::vector<float>> attentions(_layers.size());

	std::normal_distribution<float> pertDist(0.0f, _exploratoryNoise);

	for (int l = _layers.size() - 1; l >= 0; l--) {
		attentions[l].resize(_layers[l]._predictionNodes.size());

		std::vector<float> predictionActivations(_layers[l]._predictionNodes.size());
		std::vector<float> predictionStates(_layers[l]._predictionNodes.size());

		for (int pi = 0; pi < _layers[l]._predictionNodes.size(); pi++) {
			PredictionNode &p = _layers[l]._predictionNodes[pi];

			// Learn
			if (learn) {
				float predictionError = _layers[l]._sdr.getHiddenState(pi) - p._statePrev;

				float surprise = predictionError * predictionError;

				float attention = sigmoid(_layerDescs[l]._attentionFactor * (surprise - p._averageSurprise));

				attentions[l][pi] = attention;

				p._averageSurprise = (1.0f - _layerDescs[l]._averageSurpriseDecay) * p._averageSurprise + _layerDescs[l]._averageSurpriseDecay * surprise;

				if (l < _layers.size() - 1) {
					for (int ci = 0; ci < p._feedBackConnections.size(); ci++)
						p._feedBackConnections[ci]._weight += _layerDescs[l]._learnFeedBackPred * predictionError * _layers[l + 1]._predictionNodes[p._feedBackConnections[ci]._index]._stateExploratoryPrev;
				}

				// Predictive
				for (int ci = 0; ci < p._predictiveConnections.size(); ci++)
					p._predictiveConnections[ci]._weight += _layerDescs[l]._learnPredictionPred * predictionError * _layers[l]._sdr.getHiddenStatePrev(p._predictiveConnections[ci]._index);
			}

			float activation = 0.0f;

			// Feed Back
			if (l < _layers.size() - 1) {
				for (int ci = 0; ci < p._feedBackConnections.size(); ci++)
					activation += p._feedBackConnections[ci]._weight * _layers[l + 1]._predictionNodes[p._feedBackConnections[ci]._index]._stateExploratory;
			}

			// Predictive
			for (int ci = 0; ci < p._predictiveConnections.size(); ci++)
				activation += p._predictiveConnections[ci]._weight * _layers[l]._sdr.getHiddenState(p._predictiveConnections[ci]._index);

			predictionActivations[pi] = p._activation = activation;
		}

		// Inhibit to find state
		//_layers[l]._sdr.inhibit(_layerDescs[l]._sparsity, predictionActivations, predictionStates);

		for (int pi = 0; pi < _layers[l]._predictionNodes.size(); pi++) {
			PredictionNode &p = _layers[l]._predictionNodes[pi];

			p._state = predictionStates[pi];

			p._stateExploratory = std::min(1.0f, std::max(0.0f, predictionStates[pi] + pertDist(generator)));

			float error = p._stateExploratory - p._state;

			// Update traces
			if (l < _layers.size() - 1) {
				for (int ci = 0; ci < p._feedBackConnections.size(); ci++)
					p._feedBackConnections[ci]._trace = _gammaLambda * p._feedBackConnections[ci]._trace + error * _layers[l + 1]._predictionNodes[p._feedBackConnections[ci]._index]._state;
			}

			// Predictive
			for (int ci = 0; ci < p._predictiveConnections.size(); ci++)
				p._predictiveConnections[ci]._trace += _gammaLambda * p._predictiveConnections[ci]._trace + error * _layers[l]._sdr.getHiddenState(p._predictiveConnections[ci]._index);
		}
	}
	
	// Get first layer prediction
	for (int pi = 0; pi < _inputPredictionNodes.size(); pi++) {
		PredictionNode &p = _inputPredictionNodes[pi];

		// Learn
		if (learn) {
			float predictionError = _layers.front()._sdr.getVisibleState(pi) - p._statePrev;

			float surprise = predictionError * predictionError;

			for (int ci = 0; ci < p._feedBackConnections.size(); ci++)
				p._feedBackConnections[ci]._weight += _learnInputFeedBack * predictionError * _layers.front()._predictionNodes[p._feedBackConnections[ci]._index]._stateExploratoryPrev;
		}

		float activation = 0.0f;

		// Feed Back
		for (int ci = 0; ci < p._feedBackConnections.size(); ci++)
			activation += p._feedBackConnections[ci]._weight * _layers.front()._predictionNodes[p._feedBackConnections[ci]._index]._stateExploratory;

		p._state = p._activation = activation;

		p._stateExploratory = p._state + pertDist(generator);

		float error = p._stateExploratory - p._state;

		// Update traces
		for (int ci = 0; ci < p._feedBackConnections.size(); ci++)
			p._feedBackConnections[ci]._trace = _gammaLambda * p._feedBackConnections[ci]._trace + error * _inputPredictionNodes[p._feedBackConnections[ci]._index]._state;
	}

	for (int l = 0; l < _layers.size(); l++) {
		if (learn)
			_layers[l]._sdr.learn(attentions[l], _layerDescs[l]._learnFeedForward, _layerDescs[l]._learnRecurrent, _layerDescs[l]._learnLateral, _layerDescs[l]._learnThreshold, _layerDescs[l]._sparsity);

		_layers[l]._sdr.stepEnd();

		for (int pi = 0; pi < _layers[l]._predictionNodes.size(); pi++) {
			PredictionNode &p = _layers[l]._predictionNodes[pi];

			p._statePrev = p._state;
			p._stateExploratoryPrev = p._stateExploratory;
			p._activationPrev = p._activation;
		}
	}

	for (int pi = 0; pi < _inputPredictionNodes.size(); pi++) {
		PredictionNode &p = _inputPredictionNodes[pi];

		p._statePrev = p._state;
		p._stateExploratoryPrev = p._stateExploratory;
		p._activationPrev = p._activation;
	}

	float q = 0.0f;

	for (int i = 0; i < _qInputIndices.size(); i++)
		q += getAction(_qInputIndices[i]) - _qInputOffsets[i];

	q /= _qInputIndices.size();

	float tdError = reward + _gamma * q - _prevValue;

	float newQ = _prevValue + _qAlpha * tdError;

	std::cout << tdError << " " << q << std::endl;

	_prevValue = q;

	// Update predictive connections again, this time for RL
	for (int l = _layers.size() - 1; l >= 0; l--) {
		for (int pi = 0; pi < _layers[l]._predictionNodes.size(); pi++) {
			PredictionNode &p = _layers[l]._predictionNodes[pi];

			// Learn
			if (learn) {
				if (l < _layers.size() - 1) {
					for (int ci = 0; ci < p._feedBackConnections.size(); ci++) {
						p._feedBackConnections[ci]._weight += _layerDescs[l]._learnFeedBackRL * tdError * p._feedBackConnections[ci]._tracePrev;

						p._feedBackConnections[ci]._tracePrev = p._feedBackConnections[ci]._trace;
					}

					// Predictive
					for (int ci = 0; ci < p._predictiveConnections.size(); ci++) {
						p._predictiveConnections[ci]._weight += _layerDescs[l]._learnPredictionRL * tdError * p._predictiveConnections[ci]._tracePrev;

						p._predictiveConnections[ci]._tracePrev = p._predictiveConnections[ci]._trace;
					}
				}
			}
		}
	}

	// Set inputs to predictions
	for (int i = 0; i < _inputTypes.size(); i++)
		if (_inputTypes[i] == _action)
			_layers.front()._sdr.setVisibleState(i, std::min(1.0f, std::max(-1.0f, getAction(i))));
		else
			_layers.front()._sdr.setVisibleState(i, getAction(i));

	for (int i = 0; i < _qInputIndices.size(); i++)
		_layers.front()._sdr.setVisibleState(_qInputIndices[i], newQ + _qInputOffsets[i]);
}