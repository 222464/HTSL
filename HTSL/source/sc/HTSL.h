#pragma once

#include "RecurrentSparseCoder2D.h"

namespace sc {
	class HTSL {
	public:
		struct LayerDesc {
			int _width, _height;

			int _receptiveRadius;
			int _inhibitionRadius;
			int _recurrentRadius;

			int _feedbackRadius;
			int _lateralRadius;

			float _sparsity;

			float _rscAlpha;
			float _rscBeta;
			float _rscGamma;
			float _rscDelta;

			float _predictionAlpha;

			LayerDesc()
				: _width(16), _height(16),
				_receptiveRadius(8), _inhibitionRadius(6), _recurrentRadius(8),
				_feedbackRadius(8), _lateralRadius(8),
				_sparsity(0.01f), 
				_rscAlpha(0.05f), _rscBeta(0.002f), _rscGamma(0.05f), _rscDelta(40.0f),
				_predictionAlpha(0.01f)
			{}
		};
	private:
		struct PredictionConnection {
			float _weight;
			float _falloff;

			unsigned short _index;
		};

		struct PredictionNode {
			std::vector<PredictionConnection> _feedbackConnections;
			std::vector<PredictionConnection> _lateralConnections;

			float _state;

			PredictionNode()
				: _state(0.0f)
			{}
		};

		struct Layer {
			RecurrentSparseCoder2D _rsc;

			std::vector<PredictionNode> _predictionNodes;
		};

		std::vector<LayerDesc> _layerDescs;
		std::vector<Layer> _layers;

	public:
		void createRandom(int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, std::mt19937 &generator);

		void setInput(int index, float value) {
			_layers.front()._rsc.setVisibleInput(index, value);
		}

		void setInput(int x, int y, float value) {
			_layers.front()._rsc.setVisibleInput(x, y, value);
		}

		void update();
		void learnRSC();
		void learnPrediction();
		void stepEnd();

		const std::vector<LayerDesc> &getLayerDescs() const {
			return _layerDescs;
		}

		const std::vector<Layer> &getLayers() const {
			return _layers;
		}
	};
}