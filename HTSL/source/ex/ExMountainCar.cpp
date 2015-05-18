#include "ExMountainCar.h"

#include <iostream>

using namespace ex;

float mcsigmoid(float x) {
	return 1.0f / (1.0f + std::exp(-x));
}

float ExMountainCar::runStep(Agent &agent, float dt) {
	float height = (std::sin(_position * 3.0f) + 1.0f) * 0.5f;

	float reward = height * 0.1f;// _velocity > 0.0f != _prevVelocity > 0.0f ? (height > _prevHeight ? 1.0f : 0.0f) : (0.5f);

	_prevHeight = height;

	std::vector<float> input(2);

	input[0] = 0.5f * (_position + 0.52f);
	input[1] = _velocity * 15.0f;

	std::vector<float> output;

	agent.getOutput(this, input, output, reward, dt);

	float action = output[0];
	//std::cout << "A: " << action << std::endl;

	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left))
		action = -1.0f;
	else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
		action = 1.0f;

	_prevVelocity = _velocity;

	_velocity += (-_velocity * 0.01f + action * 0.001f + std::cos(3.0f * _position) * -0.0025f) * dt / 0.017f;
	_position += _velocity * dt / 0.017f;

	_prevFitness = height;

	return height;
}

void ExMountainCar::initializeVisualization() {
	_car.loadFromFile("resources/car.png");

	for (float p = -3.141596f; p < 3.141596f;) {
		sf::Vertex v1;
		v1.position = sf::Vector2f(static_cast<float>(800) * 0.5f + p * _pixelsPerMeter, static_cast<float>(600) * 0.5f + std::sin(p * 3.0f) * _pixelsPerMeter * -0.333f);
		v1.color = sf::Color::Red;

		p += 0.1f;

		sf::Vertex v2;
		v2.position = sf::Vector2f(static_cast<float>(800) * 0.5f + p * _pixelsPerMeter, static_cast<float>(600) * 0.5f + std::sin(p * 3.0f) * _pixelsPerMeter * -0.333f);
		v2.color = sf::Color::Red;

		_hills.append(v1);
		_hills.append(v2);
	}

	_hills.setPrimitiveType(sf::Lines);
}

void ExMountainCar::visualize(sf::RenderTarget &rt) const {
	rt.clear();

	rt.draw(_hills);

	sf::Sprite carSprite;
	carSprite.setTexture(_car);
	carSprite.setOrigin(sf::Vector2f(static_cast<float>(carSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(carSprite.getTexture()->getSize().y) * 0.5f));
	carSprite.setScale(sf::Vector2f(0.25f, 0.25f));

	carSprite.setPosition(sf::Vector2f(static_cast<float>(800) * 0.5f + _pixelsPerMeter * _position, static_cast<float>(600) * 0.5f + _pixelsPerMeter * -0.333f * std::sinf(3.0f * _position)));

	float slope = std::cos(3.0f * _position);

	float angle = std::atan(slope);

	carSprite.setRotation(360.0f - 180.0f / static_cast<float>(3.141596f) * angle);

	rt.draw(carSprite);
}