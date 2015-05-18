#include <Settings.h>

#if SUBPROGRAM_EXECUTE == PIANO_ROLL_PREDICTION

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <fstream>
#include <sstream>

#include <unordered_set>
#include <unordered_map>

#include <iostream>

#include <sc/HTSL.h>

struct Frame {
	std::vector<unsigned short> _notes;
};

struct Sequence {
	int _seq;
	std::vector<Frame> _frames;
};

struct Dataset {
	std::vector<Sequence> _sequences;
};

void loadDataset(const std::string &fileName, Dataset &dataset) {
	std::ifstream ff(fileName);

	while (ff.good() && !ff.eof()) {
		std::string line;

		std::getline(ff, line);

		if (line == "")
			break;

		std::istringstream fl(line);

		std::string param;

		fl >> param;
		fl >> param;

		int seq = std::stoi(param.substr(0, param.length() - 1));

		fl >> param;

		int len;

		fl >> len;

		Sequence s;

		s._seq = seq;

		s._frames.resize(len);

		for (int f = 0; f < len; f++) {
			std::getline(ff, line);

			std::istringstream fnl(line);

			while (fnl.good() && !fnl.eof()) {
				std::string noteString;
				fnl >> noteString;

				if (noteString == "" || noteString == "n")
					break;

				s._frames[f]._notes.push_back(std::stoi(noteString));
			}
		}

		dataset._sequences.push_back(s);
	}
}

int main() {
	std::mt19937 generator(time(nullptr));

	Dataset train;

	loadDataset("resources/datasets/pianorolls/piano_rolls1.txt", train);

	const int useSequence = 0;
	const int useLength = 20;

	std::unordered_set<int> usedNotes;

	for (int f = 0; f < train._sequences[useSequence]._frames.size() && f < useLength; f++) {
		Frame &frame = train._sequences[useSequence]._frames[f];

		for (int n = 0; n < frame._notes.size(); n++)
			if (usedNotes.find(frame._notes[n]) == usedNotes.end())
				usedNotes.insert(frame._notes[n]);
	}

	std::cout << "Used notes: " << usedNotes.size() << std::endl;

	std::unordered_map<int, int> noteToInput;
	std::unordered_map<int, int> inputToNote;

	int count = 0;

	for (std::unordered_set<int>::const_iterator cit = usedNotes.begin(); cit != usedNotes.end(); cit++) {
		noteToInput[*cit] = count;
		inputToNote[count] = *cit;

		count++;
	}

	sc::HTSL htsl;

	int squareDim = std::ceil(std::sqrt(static_cast<float>(usedNotes.size())));

	std::vector<sc::HTSL::LayerDesc> layerDescs(2);

	layerDescs[0]._width = 16;
	layerDescs[0]._height = 16;

	layerDescs[1]._width = 10;
	layerDescs[1]._height = 10;

	htsl.createRandom(squareDim, squareDim, layerDescs, generator);

	// Train on sequence
	for (int loop = 0; loop < 5; loop++) {
		for (int f = 0; f < train._sequences[useSequence]._frames.size() && f < useLength; f++) {
			Frame &frame = train._sequences[useSequence]._frames[f];

			for (int i = 0; i < usedNotes.size(); i++)
				htsl.setInput(i, 0.0f);

			for (int n = 0; n < frame._notes.size(); n++)
				htsl.setInput(noteToInput[frame._notes[n]], 1.0f);
			
			htsl.update();

			htsl.learn();
			htsl.stepEnd();
		}

		std::cout << "Loop " << loop << std::endl;
	}

	// Show results
	int numCorrect = 0;
	int numTotal = 0;

	std::ofstream outputFile("pianoRollOutput.txt");

	for (int f = 0; f < train._sequences[useSequence]._frames.size() && f < useLength; f++) {
		Frame &frame = train._sequences[useSequence]._frames[f];

		std::unordered_set<int> predictedNotes;

		for (int i = 0; i < squareDim * squareDim; i++)
			if (htsl.getPrediction(i) > 0.5f)
				predictedNotes.insert(inputToNote[i]);

		std::unordered_set<int> actualNotes;

		for (int n = 0; n < frame._notes.size(); n++)
			actualNotes.insert(frame._notes[n]);
		
		if (predictedNotes == actualNotes)
			numCorrect++;

		numTotal++;

		for (int i = 0; i < usedNotes.size(); i++)
			htsl.setInput(i, 0.0f);

		for (int n = 0; n < frame._notes.size(); n++)
			htsl.setInput(noteToInput[frame._notes[n]], 1.0f);

		htsl.update();

		for (int x = 0; x < layerDescs[1]._width; x++) {
			for (int y = 0; y < layerDescs[1]._height; y++) {
				std::cout << (htsl.getLayers()[1]._rsc.getHiddenBit(x, y) > 0.0f ? "1" : "0");
			}

			std::cout << std::endl;
		}

		std::cout << "Actual: ";

		for (int n = 0; n < frame._notes.size(); n++)
			std::cout << frame._notes[n] << " ";

		std::cout << "Prediction: ";

		bool noNote = true;

		for (int i = 0; i < squareDim * squareDim; i++) {
			if (htsl.getPrediction(i) > 0.5f) {
				std::cout << inputToNote[i] << " ";
				outputFile << inputToNote[i] << " ";

				noNote = false;
			}
		}

		if (noNote)
			outputFile << "n";

		std::cout << std::endl;
		outputFile << std::endl;

		htsl.stepEnd();
	}

	outputFile.close();

	std::cout << "Correct: " << (static_cast<float>(numCorrect) / static_cast<float>(numTotal) * 100.0f) << "%" << std::endl;

	return 0;
}

#endif