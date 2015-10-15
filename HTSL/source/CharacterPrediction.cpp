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
#include <hyp/BayesianOptimizer.h>

/*
void getVars(sc::HTSL &htsl, std::vector<float> &vars) {
	if (vars.size() != 13)
		vars.resize(13);

	int index = 0;

	vars[index++] = htsl.getLayerDescs()[0]._sparsity;
	vars[index++] = htsl.getLayerDescs()[0]._rscAlpha;
	vars[index++] = htsl.getLayerDescs()[0]._rscBetaVisible;
	vars[index++] = htsl.getLayerDescs()[0]._rscBetaHidden;
	vars[index++] = htsl.getLayerDescs()[0]._rscDeltaVisible;
	vars[index++] = htsl.getLayerDescs()[0]._rscDeltaHidden;
	vars[index++] = htsl.getLayerDescs()[0]._rscGamma;
	vars[index++] = htsl.getLayerDescs()[0]._nodeAlphaLateral;
	vars[index++] = htsl.getLayerDescs()[0]._nodeAlphaFeedback;
	vars[index++] = htsl.getLayerDescs()[0]._nodeBiasAlpha;
	vars[index++] = htsl.getLayerDescs()[0]._attentionAlpha;
	vars[index++] = htsl.getLayerDescs()[0]._hiddenUsageDecay;
	vars[index++] = htsl.getLayerDescs()[0]._lowUsagePreference;
}

void setVars(sc::HTSL &htsl, const std::vector<float> &vars) {
	for (int l = 0; l < htsl.getLayerDescs().size(); l++) {
		int index = 0;

		htsl.getLayerDescs()[0]._sparsity = vars[index++];
		htsl.getLayerDescs()[0]._rscAlpha = vars[index++];
		htsl.getLayerDescs()[0]._rscBetaVisible = vars[index++];
		htsl.getLayerDescs()[0]._rscBetaHidden = vars[index++];
		htsl.getLayerDescs()[0]._rscDeltaVisible = vars[index++];
		htsl.getLayerDescs()[0]._rscDeltaHidden = vars[index++];
		htsl.getLayerDescs()[0]._rscGamma = vars[index++];
		htsl.getLayerDescs()[0]._nodeAlphaLateral = vars[index++];
		htsl.getLayerDescs()[0]._nodeAlphaFeedback = vars[index++];
		htsl.getLayerDescs()[0]._nodeBiasAlpha = vars[index++];
		htsl.getLayerDescs()[0]._attentionAlpha = vars[index++];
		htsl.getLayerDescs()[0]._hiddenUsageDecay = vars[index++];
		htsl.getLayerDescs()[0]._lowUsagePreference = vars[index++];
	}
}
*/
int main() {
	std::mt19937 generator(time(nullptr));

	/*hyp::BayesianOptimizer bo;

	std::vector<float> minBounds(13, 0.0f);
	std::vector<float> maxBounds(13, 1.0f);

	bo.create(13, minBounds, maxBounds);
	bo.generateNewVariables(generator);*/

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

	std::vector<sc::HTSL::LayerDesc> layerDescs(3);

	layerDescs[0]._width = 8;
	layerDescs[0]._height = 8;

	layerDescs[1]._width = 6;
	layerDescs[1]._height = 6;

	layerDescs[2]._width = 4;
	layerDescs[2]._height = 4;

	{
		sc::HTSL htsl;

		htsl.createRandom(rootSize, rootSize, layerDescs, generator);

		// Train
		for (int iter = 0; iter < 40; iter++) {
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

			std::cout << "Train iteration " << iter << std::endl;
		}

		// Generate
		for (int i = 0; i < uniqueChars.size(); i++)
			htsl.setInput(i, 0.0f);

		htsl.setInput(charToInputIndex[text[0]], 1.0f);

		htsl.update();
		htsl.stepEnd();

		std::cout << "Example string: " << std::endl;

		std::cout << text[0];

		int numCorrect = 0;

		for (int c = 1; c < std::min(200, static_cast<int>(text.length() - 1)); c++) {
			int maxIndex = 0;

			for (int i = 1; i < uniqueChars.size(); i++)
				if (htsl.getPrediction(i) > htsl.getPrediction(maxIndex))
					maxIndex = i;

			std::cout << inputIndexToChar[maxIndex];

			if (inputIndexToChar[maxIndex] == text[c])
				numCorrect++;

			for (int i = 0; i < uniqueChars.size(); i++)
				htsl.setInput(i, 0.0f);

			htsl.setInput(maxIndex, 1.0f);

			htsl.update();
			htsl.stepEnd();
		}

		std::cout << std::endl;

		float fitness = static_cast<float>(numCorrect + 1) / static_cast<float>(text.length());

		std::cout << "Fitness: " << fitness * 100.0f << "%" << std::endl;
	}

	/*{
		sc::HTSL htsl;

		htsl.createRandom(rootSize, rootSize, layerDescs, generator);

		std::vector<float> vars;
		getVars(htsl, vars);

		bo.setCurrentVariables(vars);
	}

	float maxFitness = -99999.0f;
	std::vector<float> maxVars = bo.getCurrentVariables();

	for (int boi = 0; boi < 1000; boi++) {
		std::cout << "Bayesian optimization iteration " << boi << std::endl;

		sc::HTSL htsl;

		htsl.createRandom(rootSize, rootSize, layerDescs, generator);

		setVars(htsl, bo.getCurrentVariables());

		// Train
		for (int iter = 0; iter < 20; iter++) {
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

			std::cout << "Train iteration " << iter << std::endl;
		}

		// Generate
		for (int i = 0; i < uniqueChars.size(); i++)
			htsl.setInput(i, 0.0f);

		htsl.setInput(charToInputIndex[text[0]], 1.0f);

		htsl.update();
		htsl.stepEnd();

		std::cout << "Example string: " << std::endl;

		std::cout << text[0];

		int numCorrect = 0;

		for (int c = 1; c < text.length() - 1; c++) {
			int maxIndex = 0;

			for (int i = 1; i < uniqueChars.size(); i++)
				if (htsl.getPrediction(i) > htsl.getPrediction(maxIndex))
					maxIndex = i;

			std::cout << inputIndexToChar[maxIndex];

			if (inputIndexToChar[maxIndex] == text[c])
				numCorrect++;

			for (int i = 0; i < uniqueChars.size(); i++)
				htsl.setInput(i, 0.0f);

			htsl.setInput(maxIndex, 1.0f);

			htsl.update();
			htsl.stepEnd();
		}

		std::cout << std::endl;

		float fitness = static_cast<float>(numCorrect + 1) / static_cast<float>(text.length());

		std::cout << "Fitness: " << fitness * 100.0f << "%" << std::endl;

		std::cout << "Tested Parameters: " << std::endl;

		for (int i = 0; i < bo.getCurrentVariables().size(); i++)
			std::cout << bo.getCurrentVariables()[i] << " ";

		std::cout << std::endl;

		std::cout << "Best Parameters (fitness of " << (maxFitness * 100.0f) << "): " << std::endl;

		for (int i = 0; i < maxVars.size(); i++)
			std::cout << maxVars[i] << " ";

		std::cout << std::endl;

		if (fitness > maxFitness) {
			maxVars = bo.getCurrentVariables();
			maxFitness = fitness;
		}

		bo.update(fitness);
		bo.generateNewVariables(generator);
	}*/

	return 0;
}

#endif