#pragma once

#include "Agent.h"
#include <sc/HTSLQ.h>

namespace ex {
	class HTSLQAgent : public Agent {
	public:
		sc::HTSLQ _htslrl;

		std::mt19937 _generator;

		void initialize(int numInputs, int numOutputs) override;

		void getOutput(Experiment* pExperiment, const std::vector<float> &input, std::vector<float> &output, float reward, float dt) override;
	};
}