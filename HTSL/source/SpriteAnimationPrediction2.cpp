#include <Settings.h>

#if SUBPROGRAM_EXECUTE == SPRITE_ANIMATION_PREDICTION_2

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <fstream>
#include <sstream>

#include <unordered_set>
#include <unordered_map>

#include <iostream>

#include <sdr/PredictiveRSDR.h>

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

class FileAnimation {
public:
	std::vector<sf::Image> _images;
	std::vector<sf::Texture> _textures;

	bool loadFromFile(const std::string &root, const std::string &extension, int count) {
		bool loaded = true;

		_images.resize(count);
		_textures.resize(count);

		for (int i = 0; i < count; i++) {
			int places = 0;

			int n = i;

			while (n >= 10) {
				n /= 10;

				if (n > 0)
					places++;
			}

			std::string fullName = root;

			for (int p = 0; p < 4 - places; p++)
				fullName.push_back('0');

			fullName += std::to_string(i) + extension;

			if (!_images[i].loadFromFile(fullName))
				loaded = false;

			if (!_textures[i].loadFromImage(_images[i]))
				loaded = false;
		}

		return loaded;
	}
};

void generateCountSequence() {
	const int numCount = 1;
	const int width = 64;
	const int height = 48;

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

	const int frameWidth = 64;
	const int frameHeight = 48;
	const int numFrames = 90;

	FileAnimation animation;
	animation.loadFromFile("resources/rendersequence2/rendersequence_", ".png", numFrames);

	//Animation animation;
	//animation.loadFromFile("resources/numberSequence.png");

	sdr::PredictiveRSDR rsdr;

	std::vector<sdr::PredictiveRSDR::LayerDesc> layerDescs(3);

	layerDescs[0]._width = 32;
	layerDescs[0]._height = 32;

	layerDescs[1]._width = 24;
	layerDescs[1]._height = 24;

	layerDescs[2]._width = 16;
	layerDescs[2]._height = 16;

	rsdr.createRandom(frameWidth * 3, frameHeight, layerDescs, -0.001f, 0.001f, 0.001f, 0.01f, 0.1f, generator);

	// ------------------------------- Simulation Loop -------------------------------

	sf::RenderWindow renderWindow(sf::VideoMode(1280, 720), "Sprite Animation", sf::Style::Default);

	renderWindow.setFramerateLimit(20);

	bool quit = false;

	float dt = 0.017f;

	int frameCounter = 0;

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	bool change = false;

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
		//animation.getSubImage(frameWidth, frameHeight, frameCounter, subImage);

		subImage = animation._images[frameCounter];

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::P)) {
			for (int x = 0; x < frameWidth; x++)
				for (int y = 0; y < frameHeight; y++) {
					rsdr.setInput(x * 3 + 0, y, std::min(1.0f, std::max(0.0f, rsdr.getPrediction(x * 3 + 0, y))));
					rsdr.setInput(x * 3 + 1, y, std::min(1.0f, std::max(0.0f, rsdr.getPrediction(x * 3 + 1, y))));
					rsdr.setInput(x * 3 + 2, y, std::min(1.0f, std::max(0.0f, rsdr.getPrediction(x * 3 + 2, y))));
				}

			rsdr.simStep(false);
		}
		else {
			for (int x = 0; x < frameWidth; x++)
				for (int y = 0; y < frameHeight; y++) {
					rsdr.setInput(x * 3 + 0, y, std::min(1.0f, std::max(0.0f, subImage.getPixel(x, y).r / 255.0f)));
					rsdr.setInput(x * 3 + 1, y, std::min(1.0f, std::max(0.0f, subImage.getPixel(x, y).g / 255.0f)));
					rsdr.setInput(x * 3 + 2, y, std::min(1.0f, std::max(0.0f, subImage.getPixel(x, y).b / 255.0f)));
				}

			rsdr.simStep();
		}

		change = !change;

		if (!sf::Keyboard::isKeyPressed(sf::Keyboard::S)) {
			sf::Image predictionImage;
			predictionImage.create(frameWidth, frameHeight);

			for (int x = 0; x < frameWidth; x++)
				for (int y = 0; y < frameHeight; y++) {
					sf::Color c;
					c.r = 255.0f * std::min(1.0f, std::max(0.0f, rsdr.getPrediction(x * 3 + 0, y)));
					c.g = 255.0f * std::min(1.0f, std::max(0.0f, rsdr.getPrediction(x * 3 + 1, y)));
					c.b = 255.0f * std::min(1.0f, std::max(0.0f, rsdr.getPrediction(x * 3 + 2, y)));

					predictionImage.setPixel(x, y, c);
				}

			sf::Texture predictionTexture;
			predictionTexture.loadFromImage(predictionImage);

			const float scale = 3.0f;

			sf::Sprite p;
			p.setTexture(predictionTexture);
			p.setPosition(renderWindow.getSize().x - frameWidth * scale, 0.0f);

			p.setScale(scale, scale);

			renderWindow.draw(p);

			sf::Sprite s;

			s.setTexture(animation._textures[frameCounter]);
			//s.setTexture(animation._texture);
			//s.setTextureRect(animation.getSubRect(frameWidth, frameHeight, frameCounter));

			s.setScale(scale, scale);

			renderWindow.draw(s);

			float alignment = 0.0f;

			for (int l = 0; l < rsdr.getLayers().size(); l++) {
				sf::Image sdr;
				sdr.create(rsdr.getLayerDescs()[l]._width, rsdr.getLayerDescs()[l]._height);

				for (int x = 0; x < rsdr.getLayerDescs()[l]._width; x++)
					for (int y = 0; y < rsdr.getLayerDescs()[l]._height; y++) {
						sf::Color c;
						c.r = c.g = c.b = rsdr.getLayers()[l]._predictionNodes[x + y * rsdr.getLayerDescs()[l]._width]._state * 255.0f;

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
			}

			renderWindow.display();
		}

		int prevFC = frameCounter;

		frameCounter = (frameCounter + 1) % numFrames;

		if (prevFC > frameCounter)
			change = !change;
	} while (!quit);

	return 0;
}

#endif