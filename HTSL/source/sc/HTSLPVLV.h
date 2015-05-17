#pragma once

#include "HTSL.h"

#include <assert.h>

namespace sc {
	class HTSLPVLV {
	public:
		enum InputType {
			_state, _action, _pv, _lve, _lvi
		};

	private:
		struct Node {
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
		std::vector<Node> _pvNodes;
		std::vector<Node> _lveNodes;
		std::vector<Node> _lviNodes;

		float _prevValue;
		float _expectedReward;
		float _expectedSecondaryE;
		float _expectedSecondaryI;

	public:
		float _actionRandomizeChance;
		float _actionPerturbationStdDev;

		float _expectedAlpha;
		float _secondaryAlphaE;
		float _secondaryAlphaI;

		float _thetaMin;
		float _thetaMax;

		HTSLPVLV()
			: _prevValue(0.0f), _expectedReward(0.0f), _expectedSecondaryE(0.0f), _expectedSecondaryI(0.0f), _actionRandomizeChance(0.05f), _actionPerturbationStdDev(0.05f),
			_expectedAlpha(1.0f), _secondaryAlphaE(1.0f), _secondaryAlphaI(0.1f), _thetaMin(0.2f), _thetaMax(0.8f)
		{}

		void createRandom(int inputWidth, int inputHeight, const std::vector<InputType> &inputTypes, const std::vector<HTSL::LayerDesc> &layerDescs, std::mt19937 &generator);

		void setState(int index, float value) {
			assert(_inputTypes[index] == _state);

			_htsl.setInput(index, value);
		}

		int getNumActionNodes() const {
			return _actionNodes.size();
		}

		int getNumPVNodes() const {
			return _pvNodes.size();
		}

		int getNumLVENodes() const {
			return _lveNodes.size();
		}

		int getNumLVINodes() const {
			return _lviNodes.size();
		}

		float getActionFromNodeIndex(int nodeIndex) const {
			return _actionNodes[nodeIndex]._output;
		}

		float getPVFromNodeIndex(int nodeIndex) const {
			return _pvNodes[nodeIndex]._state;
		}

		float getLVEFromNodeIndex(int nodeIndex) const {
			return _lveNodes[nodeIndex]._state;
		}

		float getLVIFromNodeIndex(int nodeIndex) const {
			return _lviNodes[nodeIndex]._state;
		}

		void update(float reward, std::mt19937 &generator);

		HTSL &getHTSL() {
			return _htsl;
		}
	};
}