#include <Settings.h>

#if SUBPROGRAM_EXECUTE == SPRITE_ANIMATION_PREDICTION

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <fstream>
#include <sstream>

#include <unordered_set>
#include <unordered_map>

#include <iostream>

#include <sc/HTSL.h>

class Animation {
public:
	sf::Image _image;
	sf::Texture _texture;

	bool loadFromFile(const std::string &fileName) {
		bool loaded = true;

		if (!_image.loadFromFile(fileName))
			loaded = false;

		if (!_texture.loadFromImage(_image))
			loaded = false;

		return loaded; 
	}

	void getSubImage(int frameWidth, int frameHeight, int frame, sf::Image &image) const {
		image.create(frameWidth, frameHeight);

		int framesInX = _image.getSize().x / frameWidth;

		int x = frame % framesInX;
		int y = frame / framesInX;

		image.copy(_image, 0, 0, sf::IntRect(x * frameWidth, y * frameHeight, frameWidth, frameHeight));
	}

	sf::IntRect getSubRect(int frameWidth, int frameHeight, int frame) const {
		int framesInX = _image.getSize().x / frameWidth;

		int x = frame % framesInX;
		int y = frame / framesInX;
		
		return sf::IntRect(x * frameWidth, y * frameHeight, frameWidth, frameHeight);
	}
};

int main() {
	std::mt19937 generator(time(nullptr));

	const int frameWidth = 34;
	const int frameHeight = 60;
	const int numFrames = 50;

	Animation animation;
	animation.loadFromFile("resources/animation.png");

	sc::HTSL htsl;

	std::vector<sc::HTSL::LayerDesc> layerDescs(3);

	layerDescs[0]._width = 64;
	layerDescs[0]._height = 64;

	layerDescs[1]._width = 48;
	layerDescs[1]._height = 48;

	layerDescs[2]._width = 32;
	layerDescs[2]._height = 32;

	htsl.createRandom(frameWidth, frameHeight, layerDescs, generator);

	// ------------------------------- Simulation Loop -------------------------------

	sf::RenderWindow renderWindow(sf::VideoMode(1280, 720), "Sprite Animation", sf::Style::Default);

	renderWindow.setFramerateLimit(40);

	bool quit = false;

	float dt = 0.017f;

	int frameCounter = 1;

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

		sf::Image subImage;
		animation.getSubImage(frameWidth, frameHeight, frameCounter, subImage);

		for (int x = 0; x < frameWidth; x++)
			for (int y = 0; y < frameHeight; y++) {
				htsl.setInput(x, y, (subImage.getPixel(x, y).r + subImage.getPixel(x, y).g + subImage.getPixel(x, y).b) * 0.333f / 255.0f);
			}

		htsl.update();
		htsl.learnRSC();
		htsl.learnPrediction();
		htsl.stepEnd();

		sf::Image predictionImage;
		predictionImage.create(frameWidth, frameHeight);

		for (int x = 0; x < frameWidth; x++)
			for (int y = 0; y < frameHeight; y++) {
				sf::Color c;
				c.r = c.g = c.b = 255.0f * htsl.getPrediction(x, y);

				predictionImage.setPixel(x, y, c);
			}

		sf::Texture predictionTexture;
		predictionTexture.loadFromImage(predictionImage);

		const float scale = 4.0f;

		sf::Sprite p;
		p.setTexture(predictionTexture);
		p.setPosition(renderWindow.getSize().x - frameWidth * scale, 0.0f);

		p.setScale(scale, scale);

		renderWindow.draw(p);

		sf::Sprite s;

		s.setTexture(animation._texture);
		s.setTextureRect(animation.getSubRect(frameWidth, frameHeight, frameCounter));

		s.setScale(scale, scale);

		renderWindow.draw(s);

		frameCounter = (frameCounter + 1) % numFrames;

		renderWindow.display();
	} while (!quit);

	return 0;
}

#endif