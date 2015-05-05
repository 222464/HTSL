#include <Settings.h>

#if SUBPROGRAM_EXECUTE == REINFORCEMENT_LEARNING

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <ex/Agent.h>
#include <ex/RandomAgent.h>
#include <ex/HTSLRLAgent.h>
#include <ex/Experiment.h>
#include <ex/ExPoleBalancing.h>

#include <vis/Plot.h>

int main() {
	sf::RenderWindow renderWindow;

	renderWindow.create(sf::VideoMode(1600, 600), "Reinforcement Learning", sf::Style::Default);

	renderWindow.setVerticalSyncEnabled(true);
	renderWindow.setFramerateLimit(60);

	// ---------------------------------- RL Init ------------------------------------

	// Change experiments and agents here!
	ex::ExPoleBalancing poleBalancing;

	poleBalancing.initializeVisualization();

	ex::HTSLRLAgent agent;

	agent.initialize(poleBalancing.getNumInputs(), poleBalancing.getNumOutputs());

	// ---------------------------------- Plotting -----------------------------------

	vis::Plot plot;

	plot._curves.resize(1);

	sf::RenderTexture plotRT;
	plotRT.create(800, 600, false);

	sf::Texture lineGradient;
	lineGradient.loadFromFile("resources/lineGradient.png");

	sf::Font tickFont;
	tickFont.loadFromFile("resources/arial.ttf");

	const int plotSampleTicks = 60;
	int plotSampleCounter = 0;

	const float avgDecay = 0.1f;
	float avgReward = 0.0f;

	float minReward = 0.0f;
	float maxReward = 0.001f;

	// ------------------------------- Simulation Loop -------------------------------

	bool quit = false;

	float dt = 0.017f;

	do {
		sf::Event event;

		while (renderWindow.pollEvent(event)) {
			switch (event.type) {
			case sf::Event::Closed:
				quit = true;
				break;
			}
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			quit = true;

		renderWindow.clear();

		float reward = poleBalancing.runStep(agent, dt);

		minReward = std::min(minReward, reward);
		maxReward = std::max(maxReward, reward);

		avgReward = (1.0f - avgDecay * dt) * avgReward + avgDecay * reward * dt;

		if (plotSampleCounter == plotSampleTicks) {
			plotSampleCounter = 0;

			vis::Point p;
			p._position.x = plot._curves[0]._points.size() - 1;
			p._position.y = avgReward;
			p._color = sf::Color::Red;

			plot._curves[0]._points.push_back(p);
		}

		plotSampleCounter++;

		poleBalancing.visualize(renderWindow);

		plotRT.setActive();
		plotRT.clear(sf::Color::White);

		plot.draw(plotRT, lineGradient, tickFont, 0.5f, sf::Vector2f(0.0f, plot._curves[0]._points.size()), sf::Vector2f(minReward, maxReward), sf::Vector2f(64.0f, 64.0f), sf::Vector2f(plot._curves[0]._points.size() / 10.0f, (maxReward - minReward) / 10.0f), 2.0f, 4.0f, 2.0f, 6.0f, 2.0f, 4);

		plotRT.display();

		sf::Sprite plotSprite;
		plotSprite.setTexture(plotRT.getTexture());

		plotSprite.setPosition(800.0f, 0.0f);

		renderWindow.draw(plotSprite);

		renderWindow.display();
	} while (!quit);

	return 0;
}

#endif