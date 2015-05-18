#include "HTSLRLAgent.h"

using namespace ex;

const int numQValues = 6;

void HTSLRLAgent::initialize(int numInputs, int numOutputs) {
	_generator.seed(time(nullptr));

	int totalState = numInputs + numOutputs + numQValues;

	int rootDim = std::ceil(std::sqrt(static_cast<float>(totalState)));

	std::vector<sc::HTSLSARSA::InputType> inputTypes(rootDim * rootDim, sc::HTSLSARSA::_state);

	for (int i = 0; i < inputTypes.size(); i++) {
		if (i < numInputs)
			inputTypes[i] = sc::HTSLSARSA::_state;
		else if (i < numInputs + numOutputs)
			inputTypes[i] = sc::HTSLSARSA::_action;
		else if (i < numInputs + numOutputs + numQValues)
			inputTypes[i] = sc::HTSLSARSA::_q;
	}

	std::vector<sc::HTSL::LayerDesc> layerDescs(2);

	layerDescs[0]._width = 22;
	layerDescs[0]._height = 22;

	layerDescs[1]._width = 14;
	layerDescs[1]._height = 14;

	_htslrl.createRandom(rootDim, rootDim, 12, inputTypes, layerDescs, _generator);
}

void HTSLRLAgent::getOutput(Experiment* pExperiment, const std::vector<float> &input, std::vector<float> &output, float reward, float dt) {
	int inputIndex = 0;

	for (int i = 0; i < input.size(); i++)
		_htslrl.setState(inputIndex++, input[i]);

	_htslrl.update(reward, _generator);

	output.resize(_htslrl.getNumActionNodes());

	for (int i = 0; i < output.size(); i++)
		output[i] = _htslrl.getActionFromNodeIndex(i) * 2.0f - 1.0f;
}