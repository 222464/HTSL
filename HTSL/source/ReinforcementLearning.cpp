#include <Settings.h>

#if SUBPROGRAM_EXECUTE == REINFORCEMENT_LEARNING

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <ex/Agent.h>
#include <ex/RandomAgent.h>
#include <ex/HTSLSARSAAgent.h>
#include <ex/HTSLQAgent.h>
#include <ex/SOUAgent.h>
#include <ex/Experiment.h>
#include <ex/ExPoleBalancing.h>
#include <ex/ExMountainCar.h>

#include <vis/Plot.h>

int main() {
	sf::RenderWindow renderWindow;

	renderWindow.create(sf::VideoMode(1600, 600), "Reinforcement Learning", sf::Style::Default);

	renderWindow.setVerticalSyncEnabled(true);
	renderWindow.setFramerateLimit(60);

	// ---------------------------------- RL Init ------------------------------------

	// Change experiments and agents here!
	ex::ExMountainCar experiment;

	experiment.initializeVisualization();

	ex::SOUAgent agent;

	agent.initialize(experiment.getNumInputs(), experiment.getNumOutputs());

	// ---------------------------------- Plotting -----------------------------------

	vis::Plot plot;

	plot._curves.resize(1);

	sf::RenderTexture plotRT;
	plotRT.create(800, 600, false);

	sf::Texture lineGradient;
	lineGradient.loadFromFile("resources/lineGradient.png");

	sf::Font tickFont;
	tickFont.loadFromFile("resources/arial.ttf");

	const int plotSampleTicks = 300;
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

		float reward = experiment.runStep(agent, dt);

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

		if (!sf::Keyboard::isKeyPressed(sf::Keyboard::K)) {
			experiment.visualize(renderWindow);

			plotRT.setActive();
			plotRT.clear(sf::Color::White);

			plot.draw(plotRT, lineGradient, tickFont, 0.5f, sf::Vector2f(0.0f, plot._curves[0]._points.size()), sf::Vector2f(minReward, maxReward), sf::Vector2f(64.0f, 64.0f), sf::Vector2f(plot._curves[0]._points.size() / 10.0f, (maxReward - minReward) / 10.0f), 2.0f, 4.0f, 2.0f, 6.0f, 2.0f, 4);

			plotRT.display();

			sf::Sprite plotSprite;
			plotSprite.setTexture(plotRT.getTexture());

			plotSprite.setPosition(800.0f, 0.0f);

			renderWindow.draw(plotSprite);

			/*float scale = 2.0f;

			float alignment = 0.0f;

			for (int l = 0; l < agent._htslrl.getHTSL().getLayers().size(); l++) {
			sf::Image sdr;
			sdr.create(agent._htslrl.getHTSL().getLayerDescs()[l]._width, agent._htslrl.getHTSL().getLayerDescs()[l]._height);

			for (int x = 0; x < agent._htslrl.getHTSL().getLayerDescs()[l]._width; x++)
			for (int y = 0; y < agent._htslrl.getHTSL().getLayerDescs()[l]._height; y++) {
			sf::Color c;
			c.r = c.g = c.b = agent._htslrl.getHTSL().getLayers()[l]._rsc.getHiddenState(x, y) * 255.0f;

			sdr.setPixel(x, y, c);
			}

			sf::Texture sdrt;
			sdrt.loadFromImage(sdr);

			sf::Sprite sdrs;
			sdrs.setTexture(sdrt);
			sdrs.setPosition(alignment, renderWindow.getSize().y - sdr.getSize().y * scale);
			sdrs.setScale(scale, scale);

			renderWindow.draw(sdrs);

			alignment += scale * sdr.getSize().x;
			}*/

			renderWindow.display();
		}
	} while (!quit);

	return 0;
}

#endif