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
			float _rscBetaVisible;
			float _rscBetaHidden;
			float _rscGamma;

			float _predictionAlpha;

			LayerDesc()
				: _width(16), _height(16),
				_receptiveRadius(5), _inhibitionRadius(5), _recurrentRadius(5),
				_feedbackRadius(5), _lateralRadius(5),
				_sparsity(3.0f / 121.0f), 
				_rscAlpha(0.05f), _rscBetaVisible(0.01f), _rscBetaHidden(0.01f), _rscGamma(0.05f),
				_predictionAlpha(0.05f)
			{}
		};

		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

	private:
		struct PredictionConnection {
			float _weight;
			float _falloff;

			unsigned short _index;
		};

		struct PredictionNode {
			std::vector<PredictionConnection> _feedbackConnections;
			std::vector<PredictionConnection> _lateralConnections;

			float _bias;

			float _state;
			float _statePrev;

			PredictionNode()
				: _state(0.0f), _statePrev(0.0f), _bias(0.0f)
			{}
		};

		struct Layer {
			RecurrentSparseCoder2D _rsc;

			std::vector<PredictionNode> _predictionNodes;
		};

		std::vector<LayerDesc> _layerDescs;
		std::vector<Layer> _layers;

		int _inputWidth, _inputHeight;

	public:
		void createRandom(int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, std::mt19937 &generator);

		void setInput(int index, float value) {
			_layers.front()._rsc.setVisibleInput(index, value);
		}

		void setInput(int x, int y, float value) {
			_layers.front()._rsc.setVisibleInput(x, y, value);
		}

		float getPrediction(int index) const {
			return _layers.front()._predictionNodes[index]._state;
		}

		float getPrediction(int x, int y) const {
			return _layers.front()._predictionNodes[x + y * _inputWidth]._state;
		}

		void update();
		void updateUnboundedFirstLayer();
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