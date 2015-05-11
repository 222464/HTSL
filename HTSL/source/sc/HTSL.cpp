#include "HTSL.h"

#include <algorithm>

#include <iostream>

using namespace sc;

void HTSL::createRandom(int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, std::mt19937 &generator) {
	_inputWidth = inputWidth;
	_inputHeight = inputHeight;

	_layerDescs = layerDescs;

	_layers.resize(layerDescs.size());

	int prevWidth = inputWidth;
	int prevHeight = inputHeight;

	std::uniform_real_distribution<float> weightDist(0.0f, 1.0f);

	for (int l = 0; l < _layers.size(); l++) {
		float visibleToHiddenWidth = static_cast<float>(_layerDescs[l]._width) / static_cast<float>(prevWidth);
		float visibleToHiddenHeight = static_cast<float>(_layerDescs[l]._height) / static_cast<float>(prevHeight);

		_layers[l]._rsc.createRandom(prevWidth, prevHeight, _layerDescs[l]._width, _layerDescs[l]._height,
			_layerDescs[l]._receptiveRadius, _layerDescs[l]._inhibitionRadius, _layerDescs[l]._recurrentRadius, generator);

		_layers[l]._predictionGroups.resize(prevWidth * prevHeight);

		int lateralSize = std::pow(_layerDescs[l]._lateralRadius * 2 + 1, 2);
		int feedbackSize = std::pow(_layerDescs[l]._feedbackRadius * 2 + 1, 2);

		for (int vi = 0; vi < _layers[l]._predictionGroups.size(); vi++) {
			int vx = vi % prevWidth;
			int vy = vi / prevWidth;

			int centerX = std::round(vx * visibleToHiddenWidth);
			int centerY = std::round(vy * visibleToHiddenHeight);

			_layers[l]._predictionGroups[vi]._predictionNodes.resize(_layerDescs[l]._predictionGroupSize);

			//float groupDist2 = 0.0f;

			for (int gi = 0; gi < _layerDescs[l]._predictionGroupSize; gi++) {
				PredictionNode &node = _layers[l]._predictionGroups[vi]._predictionNodes[gi];

				node._weight = weightDist(generator);

				//groupDist2 += node._weight * node._weight;

				node._lateralConnections.reserve(lateralSize);

				float dist2 = 0.0f;

				for (int dx = -_layerDescs[l]._lateralRadius; dx <= _layerDescs[l]._lateralRadius; dx++)
					for (int dy = -_layerDescs[l]._lateralRadius; dy <= _layerDescs[l]._lateralRadius; dy++) {
						int hx = centerX + dx;
						int hy = centerY + dy;

						if (hx >= 0 && hx < _layerDescs[l]._width && hy >= 0 && hy < _layerDescs[l]._height) {
							int hi = hx + hy * _layerDescs[l]._width;

							PredictionConnection c;

							c._weight = weightDist(generator) * 2.0f - 1.0f;
							c._index = hi;
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
					int centerX = std::round(vx * visibleToHiddenWidth);
					int centerY = std::round(vy * visibleToHiddenHeight);

					node._feedbackConnections.reserve(feedbackSize);

					dist2 = 0.0f;

					for (int dx = -_layerDescs[l]._feedbackRadius; dx <= _layerDescs[l]._feedbackRadius; dx++)
						for (int dy = -_layerDescs[l]._feedbackRadius; dy <= _layerDescs[l]._feedbackRadius; dy++) {
							int vox = centerX + dx;
							int voy = centerY + dy;

							if (vox >= 0 && vox < _layerDescs[l]._width && voy >= 0 && voy < _layerDescs[l]._height) {
								int voi = vox + voy * _layerDescs[l]._width;

								PredictionConnection c;

								c._weight = weightDist(generator) * 2.0f - 1.0f;
								c._index = voi;
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

			//float normFactor = 1.0f / std::sqrt(groupDist2);

			//for (int gi = 0; gi < _layerDescs[l]._predictionGroupSize; gi++)
			//	_layers[l]._predictionGroups[vi]._predictionNodes[gi]._weight *= normFactor;
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
				_layers[l]._rsc.setVisibleInput(vi, _layers[prevLayerIndex]._rsc.getHiddenBit(vi));
		}

		_layers[l]._rsc.activate();
		_layers[l]._rsc.reconstruct();
	}

	// Down (predictions)
	for (int l = _layers.size() - 1; l >= 0; l--) {
		if (l == _layers.size() - 1) {		
			for (int ni = 0; ni < _layers[l]._predictionGroups.size(); ni++) {
				float maxState = -99999.0f;
				int maxIndex = 0;

				for (int gi = 0; gi < _layerDescs[l]._predictionGroupSize; gi++) {
					PredictionNode &node = _layers[l]._predictionGroups[ni]._predictionNodes[gi];

					float sum = _layers[l]._predictionGroups[ni]._predictionNodes[gi]._bias;

					for (int ci = 0; ci < node._lateralConnections.size(); ci++)
						sum += node._lateralConnections[ci]._falloff * node._lateralConnections[ci]._weight * _layers[l]._rsc.getHiddenState(node._lateralConnections[ci]._index);

					node._state = sigmoid(sum);

					if (node._state > maxState) {
						maxState = node._state;
						maxIndex = gi;
					}
				}

				_layers[l]._predictionGroups[ni]._maxIndex = maxIndex;

				_layers[l]._predictionGroups[ni]._state = _layers[l]._predictionGroups[ni]._predictionNodes[maxIndex]._weight;
			}
		}
		else {
			for (int ni = 0; ni < _layers[l]._predictionGroups.size(); ni++) {
				float maxState = -99999.0f;
				int maxIndex = 0;

				for (int gi = 0; gi < _layerDescs[l]._predictionGroupSize; gi++) {
					PredictionNode &node = _layers[l]._predictionGroups[ni]._predictionNodes[gi];

					float sum = _layers[l]._predictionGroups[ni]._predictionNodes[gi]._bias;

					for (int ci = 0; ci < node._lateralConnections.size(); ci++)
						sum += node._lateralConnections[ci]._falloff * node._lateralConnections[ci]._weight * _layers[l]._rsc.getHiddenBit(node._lateralConnections[ci]._index);

					for (int ci = 0; ci < node._feedbackConnections.size(); ci++)
						sum += node._feedbackConnections[ci]._falloff * node._feedbackConnections[ci]._weight * _layers[l + 1]._predictionGroups[node._feedbackConnections[ci]._index]._state;

					node._state = sigmoid(sum);

					if (node._state > maxState) {
						maxState = node._state;
						maxIndex = gi;
					}
				}
				
				_layers[l]._predictionGroups[ni]._maxIndex = maxIndex;

				_layers[l]._predictionGroups[ni]._state = _layers[l]._predictionGroups[ni]._predictionNodes[maxIndex]._weight;
			}
		}
	}
}

void HTSL::learnRSC() {
	for (int l = 0; l < _layers.size(); l++)
		_layers[l]._rsc.learn(_layerDescs[l]._rscAlpha, _layerDescs[l]._rscBetaVisible, _layerDescs[l]._rscBetaHidden, _layerDescs[l]._rscGamma, _layerDescs[l]._rscDeltaVisible, _layerDescs[l]._rscDeltaHidden, _layerDescs[l]._sparsity);
}

void HTSL::learnPrediction(float importance) {
	for (int l = 0; l < _layers.size(); l++) {
		if (l == _layers.size() - 1) {
			for (int ni = 0; ni < _layers[l]._predictionGroups.size(); ni++) {
				int minErrorIndex = 0;
				float minError = std::abs(_layers[l]._rsc.getVisibleState(ni) - _layers[l]._predictionGroups[ni]._predictionNodes[0]._weight);

				for (int gi = 1; gi < _layerDescs[l]._predictionGroupSize; gi++) {
					float error = std::abs(_layers[l]._rsc.getVisibleState(ni) - _layers[l]._predictionGroups[ni]._predictionNodes[gi]._weight);

					if (error < minError) {
						minError = error;
						minErrorIndex = gi;
					}
				}

				bool correct = minErrorIndex == _layers[l]._predictionGroups[ni]._maxIndexPrev;

				if (correct) {
					_layers[l]._predictionGroups[ni]._predictionNodes[minErrorIndex]._weight += _layerDescs[l]._groupAlphaMax * (_layers[l]._rsc.getVisibleState(ni) - _layers[l]._predictionGroups[ni]._predictionNodes[minErrorIndex]._weight);

					PredictionNode &node = _layers[l]._predictionGroups[ni]._predictionNodes[minErrorIndex];

					for (int ci = 0; ci < node._lateralConnections.size(); ci++)
						node._lateralConnections[ci]._weight += importance * _layerDescs[l]._nodeAlphaLateral * _layers[l]._rsc.getHiddenBitPrev(node._lateralConnections[ci]._index);
				}
				else {
					_layers[l]._predictionGroups[ni]._predictionNodes[_layers[l]._predictionGroups[ni]._maxIndexPrev]._weight += _layerDescs[l]._groupAlphaMin * (_layers[l]._rsc.getVisibleState(ni) - _layers[l]._predictionGroups[ni]._predictionNodes[_layers[l]._predictionGroups[ni]._maxIndexPrev]._weight);

					PredictionNode &node = _layers[l]._predictionGroups[ni]._predictionNodes[_layers[l]._predictionGroups[ni]._maxIndexPrev];

					for (int ci = 0; ci < node._lateralConnections.size(); ci++)
						node._lateralConnections[ci]._weight += importance * _layerDescs[l]._nodeAlphaLateral * -_layers[l]._rsc.getHiddenBitPrev(node._lateralConnections[ci]._index);
				}

				for (int gi = 0; gi < _layerDescs[l]._predictionGroupSize; gi++)
					_layers[l]._predictionGroups[ni]._predictionNodes[gi]._bias += _layerDescs[l]._nodeBiasAlpha * (1.0f / _layerDescs[l]._predictionGroupSize - (gi == _layers[l]._predictionGroups[ni]._maxIndexPrev ? 1.0f : 0.0f));
			}
		}
		else {
			for (int ni = 0; ni < _layers[l]._predictionGroups.size(); ni++) {
				int minErrorIndex = 0;
				float minError = std::abs(_layers[l]._rsc.getVisibleState(ni) - _layers[l]._predictionGroups[ni]._predictionNodes[0]._weight);

				for (int gi = 1; gi < _layerDescs[l]._predictionGroupSize; gi++) {
					float error = std::abs(_layers[l]._rsc.getVisibleState(ni) - _layers[l]._predictionGroups[ni]._predictionNodes[gi]._weight);

					if (error < minError) {
						minError = error;
						minErrorIndex = gi;
					}
				}

				bool correct = minErrorIndex == _layers[l]._predictionGroups[ni]._maxIndexPrev;

				if (correct) {
					_layers[l]._predictionGroups[ni]._predictionNodes[minErrorIndex]._weight += _layerDescs[l]._groupAlphaMax * (_layers[l]._rsc.getVisibleState(ni) - _layers[l]._predictionGroups[ni]._predictionNodes[minErrorIndex]._weight);

					PredictionNode &node = _layers[l]._predictionGroups[ni]._predictionNodes[minErrorIndex];

					for (int ci = 0; ci < node._lateralConnections.size(); ci++)
						node._lateralConnections[ci]._weight += importance * _layerDescs[l]._nodeAlphaLateral * _layers[l]._rsc.getHiddenBitPrev(node._lateralConnections[ci]._index);
				
					for (int ci = 0; ci < node._feedbackConnections.size(); ci++)
						node._feedbackConnections[ci]._weight += importance * _layerDescs[l]._nodeAlphaFeedback * _layers[l + 1]._predictionGroups[node._feedbackConnections[ci]._index]._statePrev;
				}
				else {
					_layers[l]._predictionGroups[ni]._predictionNodes[_layers[l]._predictionGroups[ni]._maxIndexPrev]._weight += _layerDescs[l]._groupAlphaMin * (_layers[l]._rsc.getVisibleState(ni) - _layers[l]._predictionGroups[ni]._predictionNodes[_layers[l]._predictionGroups[ni]._maxIndexPrev]._weight);

					PredictionNode &node = _layers[l]._predictionGroups[ni]._predictionNodes[_layers[l]._predictionGroups[ni]._maxIndexPrev];

					for (int ci = 0; ci < node._lateralConnections.size(); ci++)
						node._lateralConnections[ci]._weight += importance * _layerDescs[l]._nodeAlphaLateral * -_layers[l]._rsc.getHiddenBitPrev(node._lateralConnections[ci]._index);
				
					for (int ci = 0; ci < node._feedbackConnections.size(); ci++)
						node._feedbackConnections[ci]._weight += importance * _layerDescs[l]._nodeAlphaFeedback * -_layers[l + 1]._predictionGroups[node._feedbackConnections[ci]._index]._statePrev;
				}

				for (int gi = 0; gi < _layerDescs[l]._predictionGroupSize; gi++)
					_layers[l]._predictionGroups[ni]._predictionNodes[gi]._bias += _layerDescs[l]._nodeBiasAlpha * (1.0f / _layerDescs[l]._predictionGroupSize - (gi == _layers[l]._predictionGroups[ni]._maxIndexPrev ? 1.0f : 0.0f));
			}
		}
	}
}

void HTSL::stepEnd() {
	for (int l = 0; l < _layers.size(); l++) {
		_layers[l]._rsc.stepEnd();

		for (int ni = 0; ni < _layers[l]._predictionGroups.size(); ni++) {
			_layers[l]._predictionGroups[ni]._statePrev = _layers[l]._predictionGroups[ni]._state;
			_layers[l]._predictionGroups[ni]._maxIndexPrev = _layers[l]._predictionGroups[ni]._maxIndex;

			for (int gi = 0; gi < _layerDescs[l]._predictionGroupSize; gi++)
				_layers[l]._predictionGroups[ni]._predictionNodes[gi]._statePrev = _layers[l]._predictionGroups[ni]._predictionNodes[gi]._state;
		}
	}
}