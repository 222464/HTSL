#include <Settings.h>

#if SUBPROGRAM_EXECUTE == BINH_TEST

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <vis/Plot.h>

#include <fstream>
#include <sstream>
#include <iostream>

#define __USE_PREDICTIVE_RSDR____

#ifndef __USE_PREDICTIVE_RSDR____
#include <sc/HTSL.h>
#else
#include <sdr/PredictiveRSDR.h>
#endif

//#define _USE_ECG_DATA

int main()
{

#ifdef _USE_ECG_DATA
	std::ifstream input("e:/ecgsyn.dat");
	float x, y, z;

#endif

	std::mt19937 generator(time(nullptr));
	sf::RenderWindow renderWindow;

	renderWindow.create(sf::VideoMode(1200, 600), "Reinforcement Learning", sf::Style::Default);

	renderWindow.setVerticalSyncEnabled(true);
	renderWindow.setFramerateLimit(60);

#ifndef __USE_PREDICTIVE_RSDR____
	sc::HTSL htsl;
#else
	sdr::PredictiveRSDR pRSDR;
#endif

#ifndef __USE_PREDICTIVE_RSDR____
	std::vector<sc::HTSL::LayerDesc> layerDescs(3);
#else
	std::vector<sdr::PredictiveRSDR::LayerDesc> layerDescs(3);
#endif
	// lowest layer (connected to sensor)
	layerDescs[0]._width  = 16;		// hidden layer width
	layerDescs[0]._height = 16;		// hidden layer height

	layerDescs[1]._width  = 8;
	layerDescs[1]._height = 8;

	// highest level
	layerDescs[2]._width  = 4;
	layerDescs[2]._height = 4;
	
	// the input images is color, that is why, we need frameWidth * 3
	//                visible_width,  visible_height, 
	// In our case, the sensor layer has the width of 96, height of 32
#ifndef __USE_PREDICTIVE_RSDR____
	htsl.createRandom(1, 1, layerDescs, generator);
#else
	float initMinWeight = -0.001f;
	float initMaxWeight = +0.001f;
	float initMinInhibition = 0.001f;
	float initMaxInhibition = 0.005f;
	pRSDR.createRandom(1, 1, layerDescs, initMinWeight, initMaxWeight, initMinInhibition, initMaxInhibition, 0.1f, generator);
#endif
	vis::Plot plot;
	plot._curves.resize(2);
	plot._curves[0]._shadow = 0.0;	// input
	plot._curves[1]._shadow = 0.0;	// predict

	sf::RenderTexture plotRT;
	plotRT.create(1200, 600, false);
	sf::Texture lineGradient;
	lineGradient.loadFromFile("resources/lineGradient.png");
	sf::Font tickFont;
	tickFont.loadFromFile("resources/arial.ttf");

	float minReward = -5.0f;
	float maxReward = +5.0f;
	plotRT.setActive();
	plotRT.clear(sf::Color::White);
	const int plotSampleTicks = 6;

	const int maxBufferSize = 500;
	bool quit = false;
	bool autoplay = false;
	float anomalyOffset = 0.f;
	float anomalyFreq = 1.f;
	float anomalyAmpl = 1.f;
	float anomalyPhase = 0.f;

	int index = -1;
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

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space))
			autoplay = false;

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::C))
			autoplay = true;
				

		if (autoplay || sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
		{
			++index;
#ifdef _USE_ECG_DATA
			if (!input.eof())
				input >> x >> y >> z;
			else
				autoplay = 0;
			// z - index of PQRSTU: P=1, Q=2, R= 3, S=4, T=5, U=6
			float value = y*4;	// without amplifying the amplitude, it is very difficult to make a sequence pattern QRST
#else
			float value = anomalyOffset + anomalyAmpl*std::sin(0.125f * 3.141596f * index * anomalyFreq + anomalyPhase) + 0.5f * std::sin(0.3f * 3.141596f * index * anomalyFreq + anomalyPhase);
#endif

#ifndef __USE_PREDICTIVE_RSDR____
			htsl.setInput(0, value);
			htsl.update();
			htsl.learn();
			htsl.stepEnd();

			float predV = htsl.getPrediction(0);
#else
			pRSDR.setInput(0, value);
			pRSDR.simStep();
			float predV = pRSDR.getPrediction(0);
#endif
			// plot target data
			vis::Point p;
			p._position.x = index;
			p._position.y = value;
			p._color = sf::Color::Red;
			plot._curves[0]._points.push_back(p);

			// plot predicted data
			vis::Point p1;
			p1._position.x = index + 1;
			p1._position.y = predV;
			p1._color = sf::Color::Blue;
			plot._curves[1]._points.push_back(p1);

			if (plot._curves[0]._points.size() > maxBufferSize)
			{
				plot._curves[0]._points.erase(plot._curves[0]._points.begin());
				int firstIndex = 0;
				for (std::vector<vis::Point>::iterator it = plot._curves[0]._points.begin(); it != plot._curves[0]._points.end(); ++it, ++firstIndex)
					(*it)._position.x = firstIndex;

				plot._curves[1]._points.erase(plot._curves[1]._points.begin());
				firstIndex = 1;
				for (std::vector<vis::Point>::iterator it = plot._curves[1]._points.begin(); it != plot._curves[1]._points.end(); ++it, ++firstIndex)
					(*it)._position.x = firstIndex;
			}

			renderWindow.clear();

			plot.draw(plotRT, lineGradient, tickFont, 0.5f, sf::Vector2f(0.0f, plot._curves[0]._points.size()), sf::Vector2f(minReward, maxReward), sf::Vector2f(64.0f, 64.0f), sf::Vector2f(plot._curves[0]._points.size() / 10.0f, (maxReward - minReward) / 10.0f), 2.0f, 4.0f, 2.0f, 6.0f, 2.0f, 4);
			plot.draw(plotRT, lineGradient, tickFont, 0.5f, sf::Vector2f(0.0f, plot._curves[1]._points.size()), sf::Vector2f(minReward, maxReward), sf::Vector2f(64.0f, 64.0f), sf::Vector2f(plot._curves[1]._points.size() / 10.0f, (maxReward - minReward) / 10.0f), 2.0f, 4.0f, 2.0f, 6.0f, 2.0f, 4);

			plotRT.display();

			sf::Sprite plotSprite;
			plotSprite.setTexture(plotRT.getTexture());

			renderWindow.draw(plotSprite);
			renderWindow.display();
		}
	} while (!quit);

	return 0;

}


#endif