#include "ExPoleBalancing.h"

using namespace ex;

float ExPoleBalancing::runStep(Agent &agent, float dt) {
	float pendulumCartAccelX = _cartAccelX;

	if (_cartX < -_cartMoveRadius)
		pendulumCartAccelX = 0.0f;
	else if (_cartX > _cartMoveRadius)
		pendulumCartAccelX = 0.0f;

	_poleAngleAccel = pendulumCartAccelX * std::cos(_poleAngle) + _g * std::sin(_poleAngle);
	_poleAngleVel += -_poleRotationalFriction * _poleAngleVel + _poleAngleAccel * dt;
	_poleAngle += _poleAngleVel * dt;

	_massPos = sf::Vector2f(_cartX + std::cos(_poleAngle + static_cast<float>(3.141596f) * 0.5f) * _poleLength, std::sin(_poleAngle + static_cast<float>(3.141596f) * 0.5f) * _poleLength);

	float reward;
	
	if (_poleAngle < static_cast<float>(3.141596f))
		reward = -(static_cast<float>(3.141596f) * 0.5f - _poleAngle);
	else
		reward = -(static_cast<float>(3.141596f) * 0.5f - (static_cast<float>(3.141596f) * 2.0f - _poleAngle));

	reward = -_cartX * 0.005f;

	float averageDecay = 0.1f;

	float force = 0.0f;

	std::vector<float> input(4, 0.0f);
	std::vector<float> output(1);

	input[0] = _cartX * 0.25f;
	input[1] = _cartVelX;
	input[2] = std::fmod(_poleAngle + static_cast<float>(3.141596f), 2.0f * static_cast<float>(3.141596f));
	input[3] = _poleAngleVel;

	agent.getOutput(this, input, output, reward, dt);

	if (std::abs(_cartVelX) < _maxSpeed)
		force = std::max(-4000.0f, std::min(4000.0f, output[0] * 4000.0f));

	if (_cartX < -_cartMoveRadius) {
		_cartX = -_cartMoveRadius;

		_cartAccelX = -_cartVelX / dt;
		_cartVelX = -0.5f * _cartVelX;
	}
	else if (_cartX > _cartMoveRadius) {
		_cartX = _cartMoveRadius;

		_cartAccelX = -_cartVelX / dt;
		_cartVelX = -0.5f * _cartVelX;
	}

	_cartAccelX = 0.25f * (force + _massMass * _poleLength * _poleAngleAccel * std::cos(_poleAngle) - _massMass * _poleLength * _poleAngleVel * _poleAngleVel * std::sin(_poleAngle)) / (_massMass + _cartMass);
	_cartVelX += -_cartFriction * _cartVelX + _cartAccelX * dt;
	_cartX += _cartVelX * dt;

	_poleAngle = std::fmod(_poleAngle, (2.0f * static_cast<float>(3.141596f)));

	if (_poleAngle < 0.0f)
		_poleAngle += static_cast<float>(3.141596f) * 2.0f;

	return reward;
}

void ExPoleBalancing::initializeVisualization() {
	_backgroundTexture.loadFromFile("resources/background.png");
	_cartTexture.loadFromFile("resources/cart.png");
	_poleTexture.loadFromFile("resources/pole.png");
}

void ExPoleBalancing::visualize(sf::RenderTarget &rt) const {
	sf::Sprite backgroundSprite;
	backgroundSprite.setTexture(_backgroundTexture);

	sf::Sprite cartSprite;
	cartSprite.setTexture(_cartTexture);

	sf::Sprite poleSprite;
	poleSprite.setTexture(_poleTexture);

	rt.draw(backgroundSprite);

	cartSprite.setOrigin(_cartTexture.getSize().x * 0.5f, _cartTexture.getSize().y * 0.5f);
	cartSprite.setPosition(sf::Vector2f(400.0f + _pixelsPerMeter * _cartX, 300.0f - 60.0f));

	rt.draw(cartSprite);

	poleSprite.setPosition(cartSprite.getPosition() + sf::Vector2f(0.0f, 17.0f));
	poleSprite.setOrigin(_poleTexture.getSize().x * 0.5f, _poleTexture.getSize().y);
	poleSprite.setRotation(_poleAngle * 180.0f / static_cast<float>(3.141596f) + 180.0f);

	rt.draw(poleSprite);
}