#pragma once

#include "Experiment.h"

namespace ex {
	class ExPoleBalancing : public Experiment {
	public:
		const float _pixelsPerMeter = 128.0f;
		const float _poleLength = 1.0f;
		const float _g = -9.8f;
		const float _massMass = 20.0f;
		const float _cartMass = 2.0f;
		const float _poleRotationalFriction = 0.008f;
		const float _cartMoveRadius = 1.8f;
		const float _cartFriction = 0.02f;
		const float _maxSpeed = 3.0f;

	private:
		sf::Vector2f _massPos;
		sf::Vector2f _massVel;
		float _poleAngle;
		float _poleAngleVel;
		float _poleAngleAccel;
		float _cartX;
		float _cartVelX;
		float _cartAccelX;

		sf::Texture _backgroundTexture;
		sf::Texture _cartTexture;
		sf::Texture _poleTexture;

		float _prevFitness;

	public:
		ExPoleBalancing()
			: _massPos(0.0f, _poleLength), _massVel(0.0f, 0.0f),
			_poleAngle(0.0f), _poleAngleVel(0.0f), _poleAngleAccel(0.0f),
			_cartX(0.0f), _cartVelX(0.0f), _cartAccelX(0.0f), _prevFitness(0.0f)
		{}

		float runStep(Agent &agent, float dt) override;

		void initializeVisualization() override;
		void visualize(sf::RenderTarget &rt) const override;

		int getNumInputs() const override {
			return 4;
		}

		int getNumOutputs() const override {
			return 1;
		}
	};
}