#include "HTSL.h"

#include <algorithm>

#include <iostream>

using namespace sc;

void HTSL::createRandom(int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, std::mt19937 &generator) {
	_inputWidth = inputWidth;
	_inputHeight = inputHeight;

	_layerDescs = layerDescs;

	_layers.resize(layerDescs.size());

	_predictedInput.clear();
	_predictedInput.assign(inputWidth * inputHeight, 0.0f);

	_predictedInputPrev.clear();
	_predictedInputPrev.assign(inputWidth * inputHeight, 0.0f);

	int prevWidth = inputWidth;
	int prevHeight = inputHeight;

	std::uniform_real_distribution<float> weightDist(0.0f, 1.0f);

	for (int l = 0; l < _layers.size(); l++) {
		_layers[l]._rsc.createRandom(prevWidth, prevHeight, _layerDescs[l]._width, _layerDescs[l]._height,
			_layerDescs[l]._receptiveRadius, _layerDescs[l]._inhibitionRadius, _layerDescs[l]._recurrentRadius, generator);

		_layers[l]._predictionNodes.resize(layerDescs[l]._width * layerDescs[l]._height);

		int lateralSize = std::pow(_layerDescs[l]._lateralRadius * 2 + 1, 2);
		int feedbackSize = std::pow(_layerDescs[l]._feedbackRadius * 2 + 1, 2);

		for (int hi = 0; hi < _layers[l]._predictionNodes.size(); hi++) {
			int hx = hi % layerDescs[l]._width;
			int hy = hi / layerDescs[l]._width;

			int centerX = hx;
			int centerY = hy;
			
			PredictionNode &node = _layers[l]._predictionNodes[hi];

			node._lateralConnections.reserve(lateralSize);

			float dist2 = 0.0f;

			for (int dx = -_layerDescs[l]._lateralRadius; dx <= _layerDescs[l]._lateralRadius; dx++)
				for (int dy = -_layerDescs[l]._lateralRadius; dy <= _layerDescs[l]._lateralRadius; dy++) {
					int hox = centerX + dx;
					int hoy = centerY + dy;

					if (hox >= 0 && hox < _layerDescs[l]._width && hoy >= 0 && hoy < _layerDescs[l]._height) {
						int hio = hox + hoy * _layerDescs[l]._width;

						PredictionConnection c;

						c._weight = weightDist(generator) * 2.0f - 1.0f;
						c._index = hio;
						c._falloff = 1.0f;// std::max(0.0f, 1.0f - std::sqrt(static_cast<float>(dx * dx + dy * dy)) / static_cast<float>(layerDescs[l]._lateralRadius + 1));

						dist2 += c._weight * c._weight;

						node._lateralConnections.push_back(c);
					}
				}

			float normFactor = 1.0f / std::sqrt(dist2);

			for (int ci = 0; ci < node._lateralConnections.size(); ci++)
				node._lateralConnections[ci]._weight *= normFactor;

			// If has another layer above it, create feedback connections
			if (l != _layers.size() - 1) {
				float hiddenToNextHiddenWidth = static_cast<float>(_layerDescs[l + 1]._width - 1) / static_cast<float>(_layerDescs[l]._width - 1);
				float hiddenToNextHiddenHeight = static_cast<float>(_layerDescs[l + 1]._height - 1) / static_cast<float>(_layerDescs[l]._height - 1);

				centerX = std::round(hiddenToNextHiddenWidth * hx);
				centerY = std::round(hiddenToNextHiddenHeight * hy);

				node._feedbackConnections.reserve(feedbackSize);

				dist2 = 0.0f;

				for (int dx = -_layerDescs[l]._feedbackRadius; dx <= _layerDescs[l]._feedbackRadius; dx++)
					for (int dy = -_layerDescs[l]._feedbackRadius; dy <= _layerDescs[l]._feedbackRadius; dy++) {
						int hox = centerX + dx;
						int hoy = centerY + dy;

						if (hox >= 0 && hox < _layerDescs[l + 1]._width && hoy >= 0 && hoy < _layerDescs[l + 1]._height) {
							int hoi = hox + hoy * _layerDescs[l + 1]._width;

							PredictionConnection c;

							c._weight = weightDist(generator) * 2.0f - 1.0f;
							c._index = hoi;
							c._falloff = 1.0f;// std::max(0.0f, 1.0f - std::sqrt(static_cast<float>(dx * dx + dy * dy)) / static_cast<float>(layerDescs[l]._feedbackRadius + 1));

							dist2 += c._weight * c._weight;

							node._feedbackConnections.push_back(c);
						}
					}

				normFactor = 1.0f / std::sqrt(dist2);

				for (int ci = 0; ci < node._feedbackConnections.size(); ci++)
					node._feedbackConnections[ci]._weight *= normFactor;
			}
		}

		prevWidth = _layerDescs[l]._width;
		prevHeight = _layerDescs[l]._height;
	}
}

void HTSL::update() {
	// Up (feature extraction)
	for (int l = 0; l < _layers.size(); l++) {
		if (l != 0) {
			int prevLayerIndex = l - 1;

			for (int vi = 0; vi < _layers[prevLayerIndex]._rsc.getNumHidden(); vi++)
				_layers[l]._rsc.setVisibleInput(vi, _layers[prevLayerIndex]._rsc.getHiddenState(vi));
		}

		_layers[l]._rsc.activate();
		_layers[l]._rsc.reconstruct();
	}

	// Down (predictions)
	for (int l = _layers.size() - 1; l >= 0; l--) {
		if (l == _layers.size() - 1) {
			// Activations
			for (int ni = 0; ni < _layers[l]._predictionNodes.size(); ni++) {
				PredictionNode &node = _layers[l]._predictionNodes[ni];

				float sum = node._bias;

				for (int ci = 0; ci < node._lateralConnections.size(); ci++)
					sum += node._lateralConnections[ci]._falloff * node._lateralConnections[ci]._weight * _layers[l]._rsc.getHiddenState(node._lateralConnections[ci]._index);

				node._activation = sum;
			}
		}
		else {
			// Activations
			for (int ni = 0; ni < _layers[l]._predictionNodes.size(); ni++) {
				PredictionNode &node = _layers[l]._predictionNodes[ni];

				float sum = node._bias;

				for (int ci = 0; ci < node._lateralConnections.size(); ci++)
					sum += node._lateralConnections[ci]._falloff * node._lateralConnections[ci]._weight * _layers[l]._rsc.getHiddenState(node._lateralConnections[ci]._index);

				for (int ci = 0; ci < node._feedbackConnections.size(); ci++)
					sum += node._feedbackConnections[ci]._falloff * node._feedbackConnections[ci]._weight * _layers[l + 1]._predictionNodes[node._feedbackConnections[ci]._index]._bit;

				node._activation = sum;
			}
		}

		// Inhibition
		for (int ni = 0; ni < _layers[l]._predictionNodes.size(); ni++) {
			const RecurrentSparseCoder2D &rsc = _layers[l]._rsc;

			PredictionNode &node = _layers[l]._predictionNodes[ni];

			float inhibition = 0.0f;

			for (int ci = 0; ci < rsc._hidden[ni]._hiddenHiddenConnections.size(); ci++)
				inhibition += rsc._hidden[ni]._hiddenHiddenConnections[ci]._weight * rsc._hidden[ni]._hiddenHiddenConnections[ci]._falloff * (_layers[l]._predictionNodes[rsc._hidden[ni]._hiddenHiddenConnections[ci]._index]._activation > node._activation ? 1.0f : 0.0f);

			node._state = std::max(0.0f, sigmoid(node._activation + inhibition) * 2.0f - 1.0f);

			node._bit = node._state > 0.0f ? 1.0f : 0.0f;
		}
	}

	// Reconstruct input
	for (int vi = 0; vi < _predictedInput.size(); vi++)
		_predictedInput[vi] = 0.0f;

	std::vector<float> sums(_predictedInput.size(), 0.0f);

	for (int hi = 0; hi < _layers.front()._predictionNodes.size(); hi++) {
		for (int ci = 0; ci < _layers.front()._rsc._hidden[hi]._visibleHiddenConnections.size(); ci++) {
			_predictedInput[_layers.front()._rsc._hidden[hi]._visibleHiddenConnections[ci]._index] += _layers.front()._rsc._hidden[hi]._visibleHiddenConnections[ci]._weight * _layers.front()._predictionNodes[hi]._bit;

			sums[_layers.front()._rsc._hidden[hi]._visibleHiddenConnections[ci]._index] += _layers.front()._predictionNodes[hi]._bit;
		}
	}

	for (int vi = 0; vi < _predictedInput.size(); vi++)
		_predictedInput[vi] /= std::max(0.0001f, sums[vi]);
}

void HTSL::learn(float importance) {
	//float predictedInputError = 0.0f;

	//for (int i = 0; i < _predictedInputPrev.size(); i++)
	//	predictedInputError += std::pow(_layers.front()._rsc.getVisibleState(i) - _predictedInputPrev[i], 2);

	for (int l = 0; l < _layers.size(); l++) {
		for (int ni = 0; ni < _layers[l]._predictionNodes.size(); ni++) {
			PredictionNode &node = _layers[l]._predictionNodes[ni];

			node._error = _layers[l]._rsc.getHiddenState(ni) - node._bitPrev;
		}
	}

	for (int l = 0; l < _layers.size(); l++) {
		if (l == _layers.size() - 1) {
			for (int ni = 0; ni < _layers[l]._predictionNodes.size(); ni++) {
				PredictionNode &node = _layers[l]._predictionNodes[ni];

				node._bias += _layerDescs[l]._nodeBiasAlpha * node._error;

				for (int ci = 0; ci < node._lateralConnections.size(); ci++)
					node._lateralConnections[ci]._weight += _layerDescs[l]._nodeAlphaLateral * node._error * _layers[l]._rsc.getHiddenStatePrev(node._lateralConnections[ci]._index);
			}
		}
		else {
			for (int ni = 0; ni < _layers[l]._predictionNodes.size(); ni++) {
				PredictionNode &node = _layers[l]._predictionNodes[ni];

				node._bias += _layerDescs[l]._nodeBiasAlpha * node._error;

				for (int ci = 0; ci < node._lateralConnections.size(); ci++)
					node._lateralConnections[ci]._weight += _layerDescs[l]._nodeAlphaLateral * node._error * _layers[l]._rsc.getHiddenStatePrev(node._lateralConnections[ci]._index);

				for (int ci = 0; ci < node._feedbackConnections.size(); ci++)
					node._feedbackConnections[ci]._weight += _layerDescs[l]._nodeAlphaFeedback * node._error * _layers[l + 1]._predictionNodes[node._feedbackConnections[ci]._index]._bitPrev;
			}
		}
	}

	for (int l = 0; l < _layers.size(); l++)
		_layers[l]._rsc.learn(_layerDescs[l]._rscAlpha, _layerDescs[l]._rscBetaVisible, _layerDescs[l]._rscBetaHidden, _layerDescs[l]._rscGamma, _layerDescs[l]._sparsity, _layerDescs[l]._rscLearnTolerance);
}

void HTSL::stepEnd() {
	for (int l = 0; l < _layers.size(); l++) {
		_layers[l]._rsc.stepEnd();

		for (int ni = 0; ni < _layers[l]._predictionNodes.size(); ni++) {
			_layers[l]._predictionNodes[ni]._statePrev = _layers[l]._predictionNodes[ni]._state;
			_layers[l]._predictionNodes[ni]._bitPrev = _layers[l]._predictionNodes[ni]._bit;
		}
	}

	_predictedInputPrev = _predictedInput;
}