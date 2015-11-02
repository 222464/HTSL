#pragma once

#include "RSDR.h"

namespace sdr {
	class PRSDRRL {
	public:
		enum InputType {
			_state, _q, _action
		};

		struct Connection {
			unsigned short _index;

			float _weight;
			float _trace;
			float _tracePrev;

			Connection()
				: _trace(0.0f), _tracePrev(0.0f)
			{}
		};

		struct LayerDesc {
			int _width, _height;

			int _receptiveRadius, _recurrentRadius, _lateralRadius, _predictiveRadius, _feedBackRadius;

			float _learnFeedForward, _learnRecurrent, _learnLateral, _learnThreshold;

			float _learnFeedBackPred, _learnPredictionPred;
			float _learnFeedBackRL, _learnPredictionRL;

			int _subIterSettle;
			int _subIterMeasure;
			float _leak;

			float _averageSurpriseDecay;
			float _attentionFactor;

			float _sparsity;

			LayerDesc()
				: _width(16), _height(16),
				_receptiveRadius(8), _recurrentRadius(5), _lateralRadius(4), _predictiveRadius(5), _feedBackRadius(8),
				_learnFeedForward(0.05f), _learnRecurrent(0.05f), _learnLateral(0.2f), _learnThreshold(0.12f),
				_learnFeedBackPred(0.5f), _learnPredictionPred(0.5f),
				_learnFeedBackRL(0.1f), _learnPredictionRL(0.1f),
				_subIterSettle(17), _subIterMeasure(5), _leak(0.1f),
				_averageSurpriseDecay(0.01f),
				_attentionFactor(4.0f),
				_sparsity(0.02f)
			{}
		};

		struct PredictionNode {
			std::vector<Connection> _feedBackConnections;
			std::vector<Connection> _predictiveConnections;

			Connection _bias;

			float _state;
			float _statePrev;
			
			float _stateExploratory;
			float _stateExploratoryPrev;

			float _activation;
			float _activationPrev;

			float _averageSurprise; // Use to keep track of importance for prediction. If current error is greater than average, then attention is > 0.5 else < 0.5 (sigmoid)

			PredictionNode()
				: _state(0.0f), _statePrev(0.0f), _stateExploratory(0.0f), _stateExploratoryPrev(0.0f),
				_activation(0.0f), _activationPrev(0.0f), _averageSurprise(0.0f)
			{}
		};

		struct Layer {
			RSDR _sdr;

			std::vector<PredictionNode> _predictionNodes;
		};

		std::vector<PredictionNode> _inputPredictionNodes;

		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

	private:
		std::vector<LayerDesc> _layerDescs;
		std::vector<Layer> _layers;

		std::vector<InputType> _inputTypes;

		std::vector<int> _qInputIndices;
		std::vector<float> _qInputOffsets;

		std::vector<int> _actionInputIndices;

		float _prevValue;

	public:
		float _stateLeak;
		float _exploratoryNoise;
		float _gamma;
		float _gammaLambda;
		float _actionRandomizeChance;
		float _qAlpha;
		float _learnInputFeedBack;

		PRSDRRL()
			: _prevValue(0.0f),
			_stateLeak(1.0f),
			_exploratoryNoise(0.1f),
			_gamma(0.99f),
			_gammaLambda(0.98f),
			_actionRandomizeChance(0.01f),
			_qAlpha(0.5f),
			_learnInputFeedBack(0.01f)
		{}

		void createRandom(int inputWidth, int inputHeight, int inputFeedBackRadius, const std::vector<InputType> &inputTypes, const std::vector<LayerDesc> &layerDescs, float initMinWeight, float initMaxWeight, float initThreshold, std::mt19937 &generator);

		void simStep(float reward, std::mt19937 &generator, bool learn = true);

		void setState(int index, float value) {
			_layers.front()._sdr.setVisibleState(index, value * _stateLeak + (1.0f - _stateLeak) * getAction(index));
		}

		void setState(int x, int y, float value) {
			setState(x + y * _layers.front()._sdr.getVisibleWidth(), value);
		}

		float getAction(int index) const {
			return _inputPredictionNodes[index]._stateExploratory;
		}

		float getAction(int x, int y) const {
			return getAction(x + y * _layers.front()._sdr.getVisibleWidth());
		}

		const std::vector<LayerDesc> &getLayerDescs() const {
			return _layerDescs;
		}

		const std::vector<Layer> &getLayers() const {
			return _layers;
		}
	};
}