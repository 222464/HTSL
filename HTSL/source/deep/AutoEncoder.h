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

namespace deep {
	class AutoEncoder {
	private:
		struct Neuron {
			double _bias;
			std::vector<double> _weights;
		};

		std::vector<Neuron> _hidden;
		std::vector<double> _inputBiases;
		std::vector<double> _inputErrorBuffer;

		double crossoverChooseWeight(double w1, double w2, double averageChance, std::mt19937 &generator);

	public:
		void createRandom(size_t numInputs, size_t numOutputs, double minWeight, double maxWeight, std::mt19937 &generator);
		void createFromParents(const AutoEncoder &parent1, const AutoEncoder &parent2, double averageChance, std::mt19937 &generator);
		void mutate(double perturbationChance, double perturbationStdDev, std::mt19937 &generator);

		void update(const std::vector<double> &inputs, std::vector<double> &outputs, double alpha);

		void getReconstruction(const std::vector<double> &inputs, std::vector<double> &reconstruction);
		void reconstruction(const std::vector<double> &outputs, std::vector<double> &reconstruction);

		static double sigmoid(double x) {
			return 1.0 / (1.0 + std::exp(-x));
		}

		const std::vector<double> &getInputErrorBuffer() const {
			return _inputErrorBuffer;
		}

		size_t getNumInputs() const {
			return _inputBiases.size();
		}

		size_t getNumOutputs() const {
			return _hidden.size();
		}
	};
}