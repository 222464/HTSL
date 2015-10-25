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
	class SparseCoder {
	public:
		static double sigmoid(double x) {
			return 1.0 / (1.0 + std::exp(-x));
		}

		struct VisibleConnection {
			unsigned short _index;

			double _weight;

			VisibleConnection()
			{}
		};

		struct HiddenConnection {
			unsigned short _index;

			double _weight;

			HiddenConnection()
			{}
		};

		struct HiddenNode {
			std::vector<VisibleConnection> _visibleHiddenConnections;
			std::vector<HiddenConnection> _hiddenHiddenConnections;

			double _bias;

			double _activation;
			double _state;

			HiddenNode()
				: _state(0.0), _activation(0.0), _bias(0.0)
			{}
		};

		struct VisibleNode {
			double _input;
			double _reconstruction;
			double _error;

			VisibleNode()
				: _input(0.0), _reconstruction(0.0), _error(0.0)
			{}
		};

	private:
		int _visibleSize;
		int _hiddenSize;
		int _receptiveRadius;
		int _inhibitionRadius;

		std::vector<VisibleNode> _visible;
		std::vector<HiddenNode> _hidden;

	public:
		void createRandom(int visibleSize, int hiddenSize, int receptiveRadius, int inhibitionRadius, double weightScale, std::mt19937 &generator);

		void activate();
		void reconstruct();
		void reconstruct(const std::vector<double> &hiddenStates, std::vector<double> &recon);
		void learn(double alpha, double beta, double gamma, double sparsity);

		void setVisibleInput(int index, double value) {
			_visible[index]._input = value;
		}

		double getVisibleRecon(int index) const {
			return _visible[index]._reconstruction;
		}

		double getVisibleState(int index) const {
			return _visible[index]._input;
		}

		double getHiddenState(int index) const {
			return _hidden[index]._state;
		}

		double getHiddenActivation(int index) const {
			return _hidden[index]._activation;
		}

		HiddenNode &getHiddenNode(int index) {
			return _hidden[index];
		}

		int getNumVisible() const {
			return _visible.size();
		}

		int getNumHidden() const {
			return _hidden.size();
		}

		int getVisibleSize() const {
			return _visibleSize;
		}

		int getHiddenSize() const {
			return _hiddenSize;
		}

		int getReceptiveRadius() const {
			return _receptiveRadius;
		}

		int getInhbitionRadius() const {
			return _inhibitionRadius;
		}

		double getVHWeight(int hi, int ci) const {
			return _hidden[hi]._visibleHiddenConnections[ci]._weight;
		}

		friend class HTSL;
	};
}