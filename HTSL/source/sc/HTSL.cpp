#include "HTSL.h"

#include <algorithm>

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

		_layers[l]._predictionNodes.resize(prevWidth * prevHeight);

		int lateralSize = std::pow(_layerDescs[l]._lateralRadius * 2 + 1, 2);
		int feedbackSize = std::pow(_layerDescs[l]._feedbackRadius * 2 + 1, 2);

		for (int vi = 0; vi < _layers[l]._predictionNodes.size(); vi++) {
			int vx = vi % prevWidth;
			int vy = vi / prevWidth;

			int centerX = std::round(vx * visibleToHiddenWidth);
			int centerY = std::round(vy * visibleToHiddenHeight);

			PredictionNode &node = _layers[l]._predictionNodes[vi];
				
			node._lateralConnections.reserve(lateralSize);

			float dist2 = 0.0f;

			for (int dx = -_layerDescs[l]._lateralRadius; dx <= _layerDescs[l]._lateralRadius; dx++)
				for (int dy = -_layerDescs[l]._lateralRadius; dy <= _layerDescs[l]._lateralRadius; dy++) {
					int hx = centerX + dx;
					int hy = centerY + dy;

					if (hx >= 0 && hx < _layerDescs[l]._width && hy >= 0 && hy < _layerDescs[l]._height) {
						int hi = hx + hy * _layerDescs[l]._width;

						PredictionConnection c;

						c._weight = weightDist(generator);
						c._index = hi;
						c._falloff = std::max(0.0f, 1.0f - std::sqrt(static_cast<float>(dx * dx + dy * dy)) / static_cast<float>(layerDescs[l]._lateralRadius + 1));

						dist2 += c._weight * c._weight;

						node._lateralConnections.push_back(c);
					}
				}

			float normFactor = 1.0f / dist2;

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

							c._weight = weightDist(generator);
							c._index = voi;
							c._falloff = std::max(0.0f, 1.0f - std::sqrt(static_cast<float>(dx * dx + dy * dy)) / static_cast<float>(layerDescs[l]._feedbackRadius + 1));

							dist2 += c._weight * c._weight;

							node._feedbackConnections.push_back(c);
						}
					}

				normFactor = 1.0f / dist2;

				for (int ci = 0; ci < node._feedbackConnections.size(); ci++)
					node._feedbackConnections[ci]._weight *= normFactor;
			}
		}

		prevWidth = _layerDescs[l]._width;
		prevHeight = _layerDescs[l]._height;
	}
}