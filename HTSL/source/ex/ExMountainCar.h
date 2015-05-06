#pragma once

#include "Experiment.h"

namespace ex {
	class ExMountainCar : public Experiment {
	private:
		float _velocity;
		float _position;

		float _prevFitness;

		sf::VertexArray _hills;
		sf::Texture _car;

	public:
		const float _pixelsPerMeter = 128.0f;

		ExMountainCar()
			: _velocity(0.0f), _position(-0.5f), _prevFitness(0.0f)
		{}

		float runStep(Agent &agent, float dt) override;

		void initializeVisualization() override;
		void visualize(sf::RenderTarget &rt) const override;

		int getNumInputs() const override {
			return 2;
		}

		int getNumOutputs() const override {
			return 1;
		}
	};
}