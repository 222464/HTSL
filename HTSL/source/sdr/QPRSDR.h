#pragma once

#include "PredictiveRSDR.h"

namespace sdr {
	class QPRSDR {
	public:
		struct Connection {
			unsigned short _index;

			float _weight;
			float _trace;

			Connection()
				: _trace(0.0f)
			{}
		};

		struct QFunctionNode {
			float _state;
			float _error;
			
			std::vector<Connection> _feedForwardConnections;

			QFunctionNode()
				: _state(0.0f), _error(0.0f)
			{}
		};

		struct QFunctionLayer {
			std::vector<QFunctionNode> _qFunctionNodes;
		};

		struct ActionNode {
			float _predictedAction;
			float _deriveAction;
			float _exploratoryAction;

			float _error;

			int _inputIndex;

			ActionNode()
				: _predictedAction(0.0f),
				_deriveAction(0.0f),
				_exploratoryAction(0.0f),
				_error(0.0f)
			{}
		};

	private:
		PredictiveRSDR _prsdr;

		std::vector<QFunctionLayer> _qFunctionLayers;

		std::vector<ActionNode> _actionNodes;
		std::vector<int> _actionNodeIndices;

		std::vector<Connection> _qConnections;

		float _prevValue;

	public:
		static float relu(float x, float leak) {
			return x > 0.0f ? x : x * leak;
		}

		static float relud(float x, float leak) {
			return x > 0.0f ? 1.0f : leak;
		}

		float _qAlpha;
		float _actionAlpha;
		int _actionDeriveIterations;
		float _actionDeriveAlpha;
		float _reluLeak;

		float _explorationBreak;
		float _explorationStdDev;

		float _gamma;
		float _gammaLambda;

		QPRSDR()
			: _qAlpha(0.005f),
			_actionAlpha(0.01f),
			_actionDeriveIterations(32),
			_actionDeriveAlpha(0.05f),
			_reluLeak(0.01f),
			_explorationBreak(0.01f),
			_explorationStdDev(0.05f),
			_gamma(0.99f),
			_gammaLambda(0.98f)
		{}

		void createRandom(int inputWidth, int inputHeight, const std::vector<int> &actionIndices, const std::vector<PredictiveRSDR::LayerDesc> &layerDescs, float initMinWeight, float initMaxWeight, float initThreshold, std::mt19937 &generator);

		void simStep(float reward, std::mt19937 &generator, bool learn = true);

	};
}