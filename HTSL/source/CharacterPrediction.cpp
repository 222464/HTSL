#include <Settings.h>

#if SUBPROGRAM_EXECUTE == CHARACTER_PREDICTION

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <fstream>
#include <sstream>

#include <unordered_set>
#include <unordered_map>

#include <iostream>

#include <sc/HTSL.h>

int main() {
	std::mt19937 generator(time(nullptr));

	std::ifstream fromFile("corpus.txt");

	std::string text;
	
	while (fromFile.good() && !fromFile.eof()) {
		std::string line;

		std::getline(fromFile, line);

		text += line + '\n';
	}

	// Determine unique character count
	std::unordered_set<char> uniqueChars;

	for (int i = 0; i < text.length(); i++) {
		if (uniqueChars.find(text[i]) == uniqueChars.end())
			uniqueChars.insert(text[i]);
	}

	// Mapping from characters to input indices
	std::unordered_map<char, int> charToInputIndex;
	std::vector<char> inputIndexToChar(uniqueChars.size());

	int inputIndex = 0;

	for (std::unordered_set<char>::const_iterator cit = uniqueChars.begin(); cit != uniqueChars.end(); cit++) {
		inputIndexToChar[inputIndex] = *cit;
		charToInputIndex[*cit] = inputIndex;

		inputIndex++;
	}

	int rootSize = std::ceil(std::sqrt(static_cast<float>(uniqueChars.size())));

	sc::HTSL htsl;

	std::vector<sc::HTSL::LayerDesc> layerDescs(3);

	layerDescs[0]._width = 12;
	layerDescs[0]._height = 12;

	layerDescs[1]._width = 10;
	layerDescs[1]._height = 10;

	layerDescs[2]._width = 8;
	layerDescs[2]._height = 8;

	htsl.createRandom(rootSize, rootSize, layerDescs, generator);

	// Train
	for (int iter = 0; iter < 2000; iter++) {
		for (int c = 0; c < text.length(); c++) {
			for (int i = 0; i < uniqueChars.size(); i++)
				htsl.setInput(i, 0.0f);

			htsl.setInput(charToInputIndex[text[c]], 1.0f);

			htsl.update();
			htsl.learn();
			htsl.stepEnd();

			if (c % 100 == 0)
				std::cout << "Steps: " << c << std::endl;
		}

		std::cout << "Iteration " << iter << std::endl;
	}

	// Generate
	std::cout << text[0];

	for (int i = 0; i < uniqueChars.size(); i++)
		htsl.setInput(i, 0.0f);

	htsl.setInput(charToInputIndex[text[0]], 1.0f);

	htsl.update();
	htsl.stepEnd();

	for (int c = 0; c < 100; c++) {
		float maxIndex = 0;

		for (int i = 1; i < uniqueChars.size(); i++)
			if (htsl.getPrediction(i) > htsl.getPrediction(maxIndex))
				maxIndex = i;

		std::cout << inputIndexToChar[maxIndex];

		for (int i = 0; i < uniqueChars.size(); i++)
			htsl.setInput(i, 0.0f);

		htsl.setInput(maxIndex, 1.0f);

		//for (int i = 0; i < uniqueChars.size(); i++)
		//	htsl.setInput(i, htsl.getPrediction(i));

		htsl.update();
		htsl.stepEnd();

		/*for (int x = 0; x < layerDescs[0]._width; x++) {
			for (int y = 0; y < layerDescs[0]._height; y++) {
				std::cout << (htsl.getLayers()[0]._rsc.getHiddenBit(x, y) > 0.0f ? "1" : "0");
			}

			std::cout << std::endl;
		}*/
	}

	return 0;
}

#endif