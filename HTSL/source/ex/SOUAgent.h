#pragma once

#include <ex/Agent.h>

#include <deep/SelfOptimizingUnit.h>

#include <random>

namespace ex {
	class SOUAgent : public Agent {
	private:
		int _numOutputs;

	public:
		deep::SelfOptimizingUnit _sou;

		std::mt19937 _generator;

		void initialize(int numInputs, int numOutputs) override {
			_numOutputs = numOutputs;

			_generator.seed(time(nullptr));

			_sou.createRandom(numInputs, numOutputs, 64, -0.1f, 0.1f, 0.01f, 1.0f, _generator);
		}

		void getOutput(Experiment* pExperiment, const std::vector<float> &input, std::vector<float> &output, float reward, float dt) override {
			if (output.size() != _numOutputs)
				output.resize(_numOutputs);

			for (int i = 0; i < input.size(); i++)
				_sou.setState(i, input[i]);

			_sou.simStep(reward, 0.1f, 0.994f, 0.01f, 0.2f, 0.01f, 0.1f, 0.5f, 0.99f, 0.1f, 0.05f, _generator);

			for (int i = 0; i < output.size(); i++)
				output[i] = _sou.getAction(i) * 2.0f - 1.0f;
		}
	};
}