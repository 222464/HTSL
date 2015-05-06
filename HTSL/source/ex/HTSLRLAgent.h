#pragma once

#include "Agent.h"
#include <sc/HTSLRL.h>

namespace ex {
	class HTSLRLAgent : public Agent {
	public:
		sc::HTSLRL _htslrl;
	
		std::mt19937 _generator;

		void initialize(int numInputs, int numOutputs) override;

		void getOutput(Experiment* pExperiment, const std::vector<float> &input, std::vector<float> &output, float reward, float dt) override;
	};
}