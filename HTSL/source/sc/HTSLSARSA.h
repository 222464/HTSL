#pragma once

#include "HTSL.h"

#include <assert.h>

namespace sc {
	class HTSLSARSA {
	public:
		enum InputType {
			_state, _action, _q
		};

	private:
		struct Connection {
			float _weight;
			float _trace;
			unsigned short _index;

			Connection()
				: _trace(0.0f)
			{}
		};

		struct Node {
			std::vector<Connection> _firstHiddenConnections;

			unsigned short _inputIndex;

			float _state;
			float _output;

			Node()
				: _state(0.0f), _output(0.0f)
			{}
		};

		HTSL _htsl;

		std::vector<InputType> _inputTypes;

		std::vector<Node> _actionNodes;
		std::vector<Node> _qNodes;

		int _actionQRadius;

		float _prevValue;
		float _prevNewQ;
		float _prevTdError;

	public:
		float _actionRandomizeChance;
		float _actionPerturbationStdDev;

		float _qAlpha;
		float _qGamma;
		float _qTraceDecay;

		float _actionAlpha;
		float _actionBeta;
		float _actionTraceDecay;
		float _actionTraceTemperature;

		HTSLSARSA()
			: _prevValue(0.0f), _prevNewQ(0.0f), _prevTdError(0.0f), _actionRandomizeChance(0.1f), _actionPerturbationStdDev(0.1f),
			_qAlpha(0.5f), _qGamma(0.992f), _qTraceDecay(0.01f), _actionAlpha(0.01f), _actionBeta(0.1f), _actionTraceDecay(0.01f), _actionTraceTemperature(4.0f)
		{}

		void createRandom(int inputWidth, int inputHeight, int actionQRadius, const std::vector<InputType> &inputTypes, const std::vector<HTSL::LayerDesc> &layerDescs, std::mt19937 &generator);

		void setState(int index, float value) {
			assert(_inputTypes[index] == _state);

			_htsl.setInput(index, value);
		}

		int getNumActionNodes() const {
			return _actionNodes.size();
		}

		int getNumQNodes() const {
			return _qNodes.size();
		}

		float getActionFromNodeIndex(int nodeIndex) const {
			return _actionNodes[nodeIndex]._output;
		}

		float getQFromNodeIndex(int nodeIndex) const {
			return _qNodes[nodeIndex]._state;
		}

		void update(float reward, std::mt19937 &generator);

		const HTSL &getHTSL() const {
			return _htsl;
		}
	};
}