#include <Settings.h>

#if SUBPROGRAM_EXECUTE == MOUSE_MOVEMENT_PREDICTION

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <sc/HTSL.h>

#include <vis/Plot.h>
#include <vis/PrettySDR.h>

int main() {
	std::mt19937 generator(time(nullptr));

	sf::RenderWindow renderWindow;

	sf::ContextSettings cs;
	cs.antialiasingLevel = 8;

	renderWindow.create(sf::VideoMode(800, 600), "Reinforcement Learning", sf::Style::Default, cs);
	
	renderWindow.setVerticalSyncEnabled(true);
	renderWindow.setFramerateLimit(60);

	renderWindow.setMouseCursorVisible(false);

	sc::HTSL htsl;

	std::vector<sc::HTSL::LayerDesc> layerDescs(3);

	layerDescs[0]._width = 16;
	layerDescs[0]._height = 16;

	layerDescs[1]._width = 12;
	layerDescs[1]._height = 12;

	layerDescs[2]._width = 8;
	layerDescs[2]._height = 8;

	htsl.createRandom(2, 2, layerDescs, generator);

	int ticksPerSample = 2;

	int predSteps = 1;

	int tickCounter = 0;

	std::vector<sf::Vector2f> predTrailBuffer;
	std::vector<sf::Vector2f> actualTrailBuffer;

	int trailBufferLength = 200;

	float trailDecay = 0.97f;

	predTrailBuffer.resize(trailBufferLength);
	actualTrailBuffer.resize(trailBufferLength);

	// ------------------------------- Simulation Loop -------------------------------

	bool quit = false;

	float dt = 0.017f;

	sf::Vector2i prevMousePos(-999, -999);
	sf::Vector2f predictRenderPos(-999.0f, -999.0f);

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

		sf::Vector2i mousePos = sf::Mouse::getPosition(renderWindow);

		if (prevMousePos == sf::Vector2i(-999, -999)) {
			prevMousePos = mousePos;
			predictRenderPos = sf::Vector2f(mousePos.x, mousePos.y);

			for (int t = 0; t < trailBufferLength; t++)
				predTrailBuffer[t] = actualTrailBuffer[t] = predictRenderPos;
		}

		if (tickCounter >= ticksPerSample) {
			sf::Vector2i delta = mousePos - prevMousePos;

			prevMousePos = mousePos;

			htsl.setInput(0, mousePos.x * 0.001f);
			htsl.setInput(1, mousePos.y * 0.001f);
			htsl.setInput(2, delta.x * 0.01f);
			htsl.setInput(3, delta.y * 0.01f);

			htsl.update();
			htsl.learn();
			htsl.stepEnd();

			tickCounter = 0;
		}
		else
			tickCounter++;

		sc::HTSL copy = htsl;

		for (int s = 0; s < predSteps; s++) {
			copy.setInput(0, copy.getPrediction(0));
			copy.setInput(1, copy.getPrediction(1));
			copy.setInput(2, copy.getPrediction(2));
			copy.setInput(3, copy.getPrediction(3));

			copy.update();
			copy.stepEnd();
		}

		sf::Vector2f predPos;
		predPos.x = copy.getPrediction(0) * 1000.0f;
		predPos.y = copy.getPrediction(1) * 1000.0f;

		predictRenderPos += 0.2f * (predPos - predictRenderPos);

		for (int t = trailBufferLength - 1; t > 0; t--) {
			predTrailBuffer[t] = predTrailBuffer[t - 1];
			actualTrailBuffer[t] = actualTrailBuffer[t - 1];
		}

		predTrailBuffer[0] = predictRenderPos;
		actualTrailBuffer[0] = sf::Vector2f(mousePos.x, mousePos.y);

		float intensity = 1.0f;

		sf::VertexArray predVertices;
		predVertices.setPrimitiveType(sf::LinesStrip);
		predVertices.resize(trailBufferLength);

		sf::VertexArray actualVertices;
		actualVertices.setPrimitiveType(sf::LinesStrip);
		actualVertices.resize(trailBufferLength);

		for (int t = 0; t < trailBufferLength; t++) {	
			actualVertices[t].position = actualTrailBuffer[t];
			actualVertices[t].color = sf::Color(255, 0, 0, 255 * intensity);

			predVertices[t].position = predTrailBuffer[t];
			predVertices[t].color = sf::Color(0, 255, 0, 255 * intensity);

			intensity *= trailDecay;
		}

		renderWindow.draw(actualVertices);
		renderWindow.draw(predVertices);

		float alignment = 0.0f;

		float scale = 8.0f;

		for (int l = 0; l < htsl.getLayers().size(); l++) {
			sf::Image sdr;
			sdr.create(htsl.getLayerDescs()[l]._width, htsl.getLayerDescs()[l]._height);

			for (int x = 0; x < htsl.getLayerDescs()[l]._width; x++)
				for (int y = 0; y < htsl.getLayerDescs()[l]._height; y++) {
					sf::Color c;
					c.r = c.g = c.b = htsl.getLayers()[l]._rsc.getHiddenState(x, y) * 255.0f;

					sdr.setPixel(x, y, c);
				}

			sf::Texture sdrt;
			sdrt.loadFromImage(sdr);

			alignment += scale * sdr.getSize().x;

			sf::Sprite sdrs;
			sdrs.setTexture(sdrt);
			sdrs.setPosition(renderWindow.getSize().x - alignment, renderWindow.getSize().y - sdr.getSize().y * scale);
			sdrs.setScale(scale, scale);

			vis::PrettySDR psdr;
			psdr.create(htsl.getLayerDescs()[l]._width, htsl.getLayerDescs()[l]._height);

			for (int x = 0; x < htsl.getLayerDescs()[l]._width; x++)
				for (int y = 0; y < htsl.getLayerDescs()[l]._height; y++) {
					psdr.at(x, y) = htsl.getLayers()[l]._rsc.getHiddenState(x, y);
				}

			psdr._nodeSpaceSize = scale;
			psdr.draw(renderWindow, sf::Vector2f(renderWindow.getSize().x - alignment, renderWindow.getSize().y - sdr.getSize().y * scale));

			//renderWindow.draw(sdrs);
		}

		renderWindow.display();
	} while (!quit);

	return 0;
}

#endif