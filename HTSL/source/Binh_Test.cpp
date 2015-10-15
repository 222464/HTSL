#include <Settings.h>

#if SUBPROGRAM_EXECUTE == BINH_TEST

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <vis/Plot.h>

#include <sc/HTSL.h>


int main()
{
	std::mt19937 generator(time(nullptr));
	sf::RenderWindow renderWindow;

	renderWindow.create(sf::VideoMode(800, 600), "Reinforcement Learning", sf::Style::Default);

	renderWindow.setVerticalSyncEnabled(true);
	renderWindow.setFramerateLimit(60);

	sc::HTSL htsl;

	std::vector<sc::HTSL::LayerDesc> layerDescs(3);

	// lowest layer (connected to sensor)
	layerDescs[0]._width = 32;		// hidden layer width
	layerDescs[0]._height = 32;		// hidden layer height

	layerDescs[1]._width = 16;
	layerDescs[1]._height = 16;

	// highest level
	layerDescs[2]._width = 8;
	layerDescs[2]._height = 8;

	// the input images is color, that is why, we need frameWidth * 3
	//                visible_width,  visible_height, 
	// In our case, the sensor layer has the width of 96, height of 32
	htsl.createRandom(1, 1, layerDescs, generator);

	vis::Plot plot;
	plot._curves.resize(2);
	plot._curves[0]._shadow = 0.0;	// input
	plot._curves[1]._shadow = 0.0;	// predict

	sf::RenderTexture plotRT;
	plotRT.create(800, 600, false);
	sf::Texture lineGradient;
	lineGradient.loadFromFile("resources/lineGradient.png");
	sf::Font tickFont;
	tickFont.loadFromFile("resources/arial.ttf");
	float minReward = -1.0f;
	float maxReward = +1.0f;
	plotRT.setActive();
	plotRT.clear(sf::Color::White);
	const int plotSampleTicks = 6;


	bool quit = false;


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

		int index = plot._curves[0]._points.size();
		float value = std::sin(0.125f * 3.141596f * index);

		htsl.setInput(0, value);
		htsl.update();
		htsl.learn();
		htsl.stepEnd();

		float predV = htsl.getPrediction(0);

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::S)) {
			{
				vis::Point p;
				p._position.x = index;
				p._position.y = value;
				p._color = sf::Color::Red;
				plot._curves[0]._points.push_back(p);
			}

			{
				vis::Point p;
				p._position.x = index + 1;
				p._position.y = predV;
				p._color = sf::Color::Blue;
				plot._curves[1]._points.push_back(p);
			}

			renderWindow.clear();

			plot.draw(plotRT, lineGradient, tickFont, 0.5f, sf::Vector2f(0.0f, plot._curves[0]._points.size()), sf::Vector2f(minReward, maxReward), sf::Vector2f(64.0f, 64.0f), sf::Vector2f(plot._curves[0]._points.size() / 10.0f, (maxReward - minReward) / 10.0f), 2.0f, 4.0f, 2.0f, 6.0f, 2.0f, 4);
			plot.draw(plotRT, lineGradient, tickFont, 0.5f, sf::Vector2f(0.0f, plot._curves[1]._points.size()), sf::Vector2f(minReward, maxReward), sf::Vector2f(64.0f, 64.0f), sf::Vector2f(plot._curves[1]._points.size() / 10.0f, (maxReward - minReward) / 10.0f), 2.0f, 4.0f, 2.0f, 6.0f, 2.0f, 4);

			plotRT.display();

			sf::Sprite plotSprite;
			plotSprite.setTexture(plotRT.getTexture());

			//plotSprite.setPosition(800.0f, 0.0f);

			renderWindow.draw(plotSprite);
			renderWindow.display();
		}
	} while (!quit);

	return 0;

}


#endif