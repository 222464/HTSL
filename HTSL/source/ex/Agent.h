#pragma once

#include <SFML/Graphics.hpp>

namespace ex {
	class Experiment;

	class Agent {
	private:
	public:
		virtual void initialize(int numInputs, int numOutputs) {}
		virtual void getOutput(Experiment* pExperiment, const std::vector<float> &input, std::vector<float> &output, float reward, float dt) = 0;
	};
}