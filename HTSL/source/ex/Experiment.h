#pragma once

#include "Agent.h"

namespace ex {
	class Experiment {
	public:
		virtual ~Experiment() {}

		virtual float runStep(Agent &agent, float dt) = 0;
		virtual void initializeVisualization() {}
		virtual void visualize(sf::RenderTarget &rt) const {}

		virtual int getNumInputs() const = 0;
		virtual int getNumOutputs() const = 0;
	};
}