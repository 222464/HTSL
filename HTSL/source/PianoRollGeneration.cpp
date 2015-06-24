#include <Settings.h>

#if SUBPROGRAM_EXECUTE == PIANO_ROLL_GENERATION

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

	std::vector<int> useSequences(64);

	for (int s = 0; s < useSequences.size(); s++)
		useSequences[s] = s + 1;

	const int useLength = 16;

	std::unordered_set<int> usedNotes;

	for (int s = 0; s < useSequences.size(); s++) {
		for (int f = 0; f < train._sequences[useSequences[s]]._frames.size() && f < useLength; f++) {
			Frame &frame = train._sequences[useSequences[s]]._frames[f];

			for (int n = 0; n < frame._notes.size(); n++)
				if (usedNotes.find(frame._notes[n]) == usedNotes.end())
					usedNotes.insert(frame._notes[n]);
		}
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

	std::vector<sc::HTSL::LayerDesc> layerDescs(3);

	layerDescs[0]._width = 32;
	layerDescs[0]._height = 32;

	layerDescs[1]._width = 24;
	layerDescs[1]._height = 24;

	layerDescs[2]._width = 16;
	layerDescs[2]._height = 16;

	htsl.createRandom(squareDim, squareDim, layerDescs, generator);

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	// Train on sequence
	for (int iter = 0; iter < 200; iter++) {
		int useSequence = useSequences[dist01(generator) * (useSequences.size() - 1)];

		int sf = dist01(generator) * std::min(static_cast<int>(train._sequences[useSequence]._frames.size() - 1), useLength - 1);

		for (int f = 0; f < train._sequences[useSequence]._frames.size() && f < useLength; f++) {
			Frame &frame = train._sequences[useSequence]._frames[f];

			for (int i = 0; i < usedNotes.size(); i++)
				htsl.setInput(i, -1.0f);

			for (int n = 0; n < frame._notes.size(); n++)
				htsl.setInput(noteToInput[frame._notes[n]], 1.0f);

			htsl.update();

			htsl.learn();
			htsl.stepEnd();
		}

		std::cout << "Iter " << iter << std::endl;
	}

	for (int song = 0; song < 10; song++) {
		// Activate into random state
		for (int l = 0; l < layerDescs.size(); l++) {
			for (int n = 0; n < htsl.getLayers()[l]._rsc.getNumHidden(); n++) {
				htsl.getLayers()[l]._rsc.getHiddenNode(n)._statePrev = htsl.getLayers()[l]._rsc.getHiddenNode(n)._statePrevPrev = dist01(generator) < layerDescs[l]._sparsity ? 1.0f : 0.0f;
			}
		}

			// Start at a random note
		{
			int rs = useSequences[dist01(generator) * (useSequences.size() - 1)];

			int fs = dist01(generator) * std::min(useLength - 1, static_cast<int>(train._sequences[rs]._frames.size()) - 1);

			for (int i = 0; i < usedNotes.size(); i++)
				htsl.setInput(i, -1.0f);

			for (int n = 0; n < train._sequences[rs]._frames[fs]._notes.size(); n++)
				htsl.setInput(noteToInput[train._sequences[rs]._frames[fs]._notes[n]], 1.0f);
		}

		std::ofstream outputFile("pianoRollOutput" + std::to_string(song + 1) + ".txt");

		float inputNoise = 1.4f;
		std::normal_distribution<float> distNoise(0.0f, inputNoise);

		for (int f = 0; f < 200; f++) {
			htsl.update();
			htsl.stepEnd();

			std::vector<int> predictedNotes;

			for (int i = 0; i < usedNotes.size(); i++)
				if (htsl.getPrediction(i) > 0.0f)
					predictedNotes.push_back(inputToNote[i]);

			for (int i = 0; i < usedNotes.size(); i++)
				htsl.setInput(i, std::min(1.0f, std::max(-1.0f, htsl.getPrediction(i) + distNoise(generator))));

			bool noNote = true;

			for (int i = 0; i < usedNotes.size(); i++) {
				if (htsl.getPrediction(i) > 0.0f) {
					outputFile << inputToNote[i] << " ";

					noNote = false;
				}
				//std::cout << htsl.getPrediction(i) << std::endl;
			}

			if (noNote)
				outputFile << "n";

			outputFile << std::endl;
		}

		outputFile.close();
	}

	return 0;
}

#endif