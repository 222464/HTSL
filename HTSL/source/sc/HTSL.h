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
			float _rscNoveltyPower;
			float _rscLearnTolerance;
			float _rscMinLearnTolerance;

			float _nodeAlphaLateral;
			float _nodeAlphaFeedback;

			float _nodeBiasAlpha;

			float _attentionAlpha;

			float _hiddenUsageDecay;

			float _lowUsagePreference;

			LayerDesc()
				: _width(16), _height(16),
				_receptiveRadius(6), _inhibitionRadius(6), _recurrentRadius(6),
				_feedbackRadius(6), _lateralRadius(6),
				_sparsity(10.0f / 121.0f), 
				_rscAlpha(0.2f), _rscBetaVisible(0.1f), _rscBetaHidden(0.1f), _rscGamma(0.05f), _rscNoveltyPower(0.0f), _rscLearnTolerance(0.001f), _rscMinLearnTolerance(0.0001f),
				_nodeAlphaLateral(0.1f), _nodeAlphaFeedback(0.2f), _nodeBiasAlpha(0.05f), _attentionAlpha(0.0f), _hiddenUsageDecay(0.02f), _lowUsagePreference(4.0f)
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

			float _activation;
			float _activationPrev;
			float _state;
			float _statePrev;
			float _bit;
			float _bitPrev;

			float _hiddenUsage;

			float _bias;

			float _error;

			PredictionNode()
				: _activation(0.0f), _activationPrev(0.0f), _state(0.0f), _statePrev(0.0f), _bit(0.0f), _bitPrev(0.0f), _bias(0.0f), _error(0.0f), _hiddenUsage(1.0f)
			{}
		};

		struct Layer {
			RecurrentSparseCoder2D _rsc;

			std::vector<PredictionNode> _predictionNodes;
		};

		std::vector<LayerDesc> _layerDescs;
		std::vector<Layer> _layers;

		std::vector<float> _predictedInput;
		std::vector<float> _predictedInputPrev;

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
			return _predictedInput[index];
		}

		float getPrediction(int x, int y) const {
			return _predictedInput[x + y * _inputWidth];
		}

		float getPredictionFromLayer(int l, int index) const {
			return _layers[l]._predictionNodes[index]._state;
		}

		float getPredictionFromLayer(int l, int x, int y) const {
			return _layers[l]._predictionNodes[x + y * _layerDescs[l]._width]._state;
		}

		void update();
		void learn(float importance = 1.0f);
		void stepEnd();

		const std::vector<LayerDesc> &getLayerDescs() const {
			return _layerDescs;
		}

		const std::vector<Layer> &getLayers() const {
			return _layers;
		}
	};
}