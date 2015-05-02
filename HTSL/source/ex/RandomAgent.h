#pragma once

#include <ex/Agent.h>

#include <random>

namespace ex {
	class RandomAgent : public Agent {
	private:
		int _numOutputs;

	public:
		std::mt19937 _generator;

		void initialize(int numInputs, int numOutputs) override {
			_numOutputs = numOutputs;

			_generator.seed(time(nullptr));
		}

		void getOutput(Experiment* pExperiment, const std::vector<float> &input, std::vector<float> &output, float reward, float dt) override {
			if (output.size() != _numOutputs)
				output.resize(_numOutputs);

			std::uniform_real_distribution<float> distOutput(-1.0f, 1.0f);

			for (int i = 0; i < _numOutputs; i++)
				output[i] = distOutput(_generator);
		}
	};
}