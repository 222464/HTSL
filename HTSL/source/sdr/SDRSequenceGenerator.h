#pragma once

#include "PredictiveRSDR.h"

namespace sdr {
	class SDRSequenceGenerator {
	private:
		PredictiveRSDR _predictor;

		// Both fed in as input to predictor, only the latter changes during generation
		std::vector<float> _inputSDR;
		std::vector<float> _predictionSDR;
	public:
		SDRSequenceGenerator()
		{}

		void createRandom();
	};
}