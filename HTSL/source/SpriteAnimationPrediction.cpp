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

void generateCountSequence() {
	const int numCount = 40;
	const int width = 32;
	const int height = 32;

	int root = std::ceil(std::sqrt(static_cast<float>(numCount)));

	sf::RenderTexture rt;
	rt.create(width * root, height * root);

	rt.clear();

	sf::Font font;
	font.loadFromFile("resources/arial.ttf");

	for (int i = 0; i < numCount; i++) {
		int xi = i % root;
		int yi = i / root;

		int places = 0;

		int n = i;

		while (n >= 10) {
			n /= 10;

			if (n > 0)
				places++;
		}

		std::cout << places << std::endl;
		sf::Text text;
		text.setPosition((xi + (1 - places) * 0.5f) * width, yi * height);
		text.setString(std::to_string(i));
		text.setFont(font);
		
		rt.draw(text);
	}

	rt.display();

	rt.getTexture().copyToImage().saveToFile("resources/numberSequence.png");
}

int main() {
	std::mt19937 generator(time(nullptr));

	//generateCountSequence();

	const int frameWidth = 32;
	const int frameHeight = 32;
	const int numFrames = 40;

	Animation animation;
	animation.loadFromFile("resources/numberSequence.png");

	sc::HTSL htsl;

	std::vector<sc::HTSL::LayerDesc> layerDescs(3);

	layerDescs[0]._width = 64;
	layerDescs[0]._height = 64;

	layerDescs[1]._width = 48;
	layerDescs[1]._height = 48;

	layerDescs[2]._width = 32;
	layerDescs[2]._height = 32;

	htsl.createRandom(frameWidth * 3, frameHeight, layerDescs, generator);

	// ------------------------------- Simulation Loop -------------------------------

	sf::RenderWindow renderWindow(sf::VideoMode(1280, 720), "Sprite Animation", sf::Style::Default);

	renderWindow.setFramerateLimit(10);

	bool quit = false;

	float dt = 0.017f;

	int frameCounter = 0;

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

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::P)) {
			for (int x = 0; x < frameWidth; x++)
				for (int y = 0; y < frameHeight; y++) {
					htsl.setInput(x * 3 + 0, y, htsl.getPrediction(x * 3 + 0, y));
					htsl.setInput(x * 3 + 1, y, htsl.getPrediction(x * 3 + 1, y));
					htsl.setInput(x * 3 + 2, y, htsl.getPrediction(x * 3 + 2, y));
				}

			htsl.update();
			htsl.stepEnd();
		}
		else {
			for (int x = 0; x < frameWidth; x++)
				for (int y = 0; y < frameHeight; y++) {
					htsl.setInput(x * 3 + 0, y, subImage.getPixel(x, y).r / 255.0f);
					htsl.setInput(x * 3 + 1, y, subImage.getPixel(x, y).g / 255.0f);
					htsl.setInput(x * 3 + 2, y, subImage.getPixel(x, y).b / 255.0f);
				}

			htsl.update();
			htsl.learnRSC();
			htsl.learnPrediction();
			htsl.stepEnd();
		}

		sf::Image predictionImage;
		predictionImage.create(frameWidth, frameHeight);

		for (int x = 0; x < frameWidth; x++)
			for (int y = 0; y < frameHeight; y++) {
				sf::Color c;
				c.r = 255.0f * htsl.getPrediction(x * 3 + 0, y);
				c.g = 255.0f * htsl.getPrediction(x * 3 + 1, y);
				c.b = 255.0f * htsl.getPrediction(x * 3 + 2, y);

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