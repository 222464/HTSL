/*
AI Lib
Copyright (C) 2014 Eric Laukien

This software is provided 'as-is', without any express or implied
warranty.  In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
claim that you wrote the original software. If you use this software
in a product, an acknowledgment in the product documentation would be
appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#pragma once

#include <vector>
#include <random>

namespace sc {
	class RecurrentSparseCoder2D {
	public:
		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}
	private:
		struct VisibleConnection {
			unsigned short _index;

			float _weight;

			float _falloff;

			VisibleConnection()
			{}
		};

		struct HiddenConnection {
			unsigned short _index;

			float _weight;

			float _falloff;

			HiddenConnection()
			{}
		};

		struct ReconstructionConnection {
			unsigned short _index;

			float _weight;

			ReconstructionConnection()
			{}
		};

		struct HiddenNode {
			std::vector<VisibleConnection> _visibleHiddenConnections;
			std::vector<VisibleConnection> _hiddenPrevHiddenConnections;
			std::vector<HiddenConnection> _hiddenHiddenConnections;

			float _bias;

			float _state;
			float _statePrev;
			float _statePrevPrev;
			float _bit;
			float _bitPrev;
			float _bitPrevPrev;
			float _error;
			float _activation;
			float _reconstruction; // From recurrent connections
			float _reconstructionPrev; // From recurrent connections
			float _attention;

			HiddenNode()
				: _state(0.0f), _statePrev(0.0f), _statePrevPrev(0.0f), _bit(0.0f), _bitPrev(0.0f), _bitPrevPrev(0.0f), _error(0.0f), _activation(0.0f), _reconstruction(0.0f), _attention(0.0f)
			{}
		};

		struct VisibleNode {
			float _input;
			float _inputPrev;
			float _reconstruction;
			float _reconstructionPrev;

			VisibleNode()
				: _input(0.0f), _inputPrev(0.0f), _reconstruction(0.0f), _reconstructionPrev(0.0f)
			{}
		};

		int _visibleWidth, _visibleHeight;
		int _hiddenWidth, _hiddenHeight;
		int _receptiveRadius;
		int _inhibitionRadius;
		int _recurrentRadius;

		std::vector<VisibleNode> _visible;
		std::vector<HiddenNode> _hidden;

	public:
		void createRandom(int visibleWidth, int visibleHeight, int hiddenWidth, int hiddenHeight, int receptiveRadius, int inhibitionRadius, int recurrentRadius, std::mt19937 &generator);

		void activate();
		void reconstruct();
		void learn(float alpha, float betaVisible, float betaHidden, float gamma, float sparsity, float learnTolerance);
		void attention(float alpha);
		void stepEnd();

		float getRepresentationError() const;

		void setVisibleInput(int index, float value) {
			_visible[index]._input = value;
		}

		void setVisibleInput(int x, int y, float value) {
			_visible[x + y * _visibleWidth]._input = value;
		}

		void setAttention(int index, float value) {
			_hidden[index]._attention = value;
		}

		void setAttention(int x, int y, float value) {
			_hidden[x + y * _hiddenWidth]._attention = value;
		}

		float getAttention(int index) const {
			return _hidden[index]._attention;
		}

		float getAttention(int x, int y) const {
			return _hidden[x + y * _hiddenWidth]._attention;
		}

		float getVisibleRecon(int index) const {
			return _visible[index]._reconstruction;
		}

		float getVisibleRecon(int x, int y) const {
			return _visible[x + y * _visibleWidth]._reconstruction;
		}

		float getVisibleState(int index) const {
			return _visible[index]._input;
		}

		float getVisibleState(int x, int y) const {
			return _visible[x + y * _visibleWidth]._input;
		}

		float getHiddenState(int index) const {
			return _hidden[index]._state;
		}

		float getHiddenState(int x, int y) const {
			return _hidden[x + y * _hiddenWidth]._state;
		}

		float getHiddenStatePrev(int index) const {
			return _hidden[index]._statePrev;
		}

		float getHiddenStatePrev(int x, int y) const {
			return _hidden[x + y * _hiddenWidth]._statePrev;
		}

		float getHiddenBit(int index) const {
			return _hidden[index]._bit;
		}

		float getHiddenBit(int x, int y) const {
			return _hidden[x + y * _hiddenWidth]._bit;
		}

		float getHiddenBitPrev(int index) const {
			return _hidden[index]._bitPrev;
		}

		float getHiddenBitPrev(int x, int y) const {
			return _hidden[x + y * _hiddenWidth]._bitPrev;
		}

		int getNumVisible() const {
			return _visible.size();
		}

		int getNumHidden() const {
			return _hidden.size();
		}

		int getVisibleWidth() const {
			return _visibleWidth;
		}

		int getVisibleHeight() const {
			return _visibleHeight;
		}

		int getHiddenWidth() const {
			return _hiddenWidth;
		}

		int getHiddenHeight() const {
			return _hiddenHeight;
		}

		int getReceptiveRadius() const {
			return _receptiveRadius;
		}

		int getInhbitionRadius() const {
			return _inhibitionRadius;
		}

		float getVHWeight(int hi, int ci) const {
			return _hidden[hi]._visibleHiddenConnections[ci]._weight;
		}

		float getVHWeight(int hx, int hy, int ci) const {
			return _hidden[hx + hy * _hiddenWidth]._visibleHiddenConnections[ci]._weight;
		}

		void getVHWeights(int hx, int hy, std::vector<float> &rectangle) const;

		friend class HTSL;
	};
}