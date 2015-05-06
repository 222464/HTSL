#include "HTSLRL.h"

#include <algorithm>

#include <iostream>

using namespace sc;

void HTSLRL::createRandom(int inputWidth, int inputHeight, int actionQRadius, const std::vector<InputType> &inputTypes, const std::vector<HTSL::LayerDesc> &layerDescs, std::mt19937 &generator) {
	assert(inputTypes.size() == inputWidth * inputHeight);

	_inputTypes = inputTypes;

	_actionQRadius = actionQRadius;

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	int actionQSize = std::pow(actionQRadius * 2 + 1, 2);

	float visibleToHiddenWidth = static_cast<float>(layerDescs.front()._width) / static_cast<float>(inputWidth);
	float visibleToHiddenHeight = static_cast<float>(layerDescs.front()._height) / static_cast<float>(inputHeight);

	for (int vi = 0; vi < _inputTypes.size(); vi++) {
		int vx = vi % inputWidth;
		int vy = vi / inputWidth;

		switch (_inputTypes[vi]) {
		case _state:
			break;
		case _action:
		{
			Node actionNode;

			actionNode._inputIndex = vi;
			actionNode._firstHiddenConnections.reserve(actionQSize);

			int centerX = std::round(visibleToHiddenWidth * vx);
			int centerY = std::round(visibleToHiddenHeight * vy);

			float dist2 = 0.0f;

			for (int dx = -actionQRadius; dx <= actionQRadius; dx++)
				for (int dy = -actionQRadius; dy <= actionQRadius; dy++) {
					int hx = centerX + dx;
					int hy = centerY + dy;

					if (hx >= 0 && hx < layerDescs.front()._width && hy >= 0 && hy < layerDescs.front()._height) {
						int hi = hx + hy * layerDescs.front()._width;

						{
							Connection c;

							c._weight = dist01(generator);
							c._index = hi;

							actionNode._firstHiddenConnections.push_back(c);

							dist2 += c._weight * c._weight;
						}

						// Secondary prediction connection
						{
							Connection c;

							c._weight = dist01(generator);
							c._index = hi;

							actionNode._firstHiddenConnections.push_back(c);

							dist2 += c._weight * c._weight;
						}
					}
				}

			actionNode._firstHiddenConnections.shrink_to_fit();

			float normMult = 1.0f / dist2;

			for (int ci = 0; ci < actionNode._firstHiddenConnections.size(); ci++)
				actionNode._firstHiddenConnections[ci]._weight *= normMult;

			_actionNodes.push_back(actionNode);

			break;
		}
		case _pv:
		{
			Node pvNode;

			pvNode._inputIndex = vi;
			pvNode._firstHiddenConnections.reserve(actionQSize);

			int centerX = std::round(visibleToHiddenWidth * vx);
			int centerY = std::round(visibleToHiddenHeight * vy);

			float dist2 = 0.0f;

			for (int dx = -actionQRadius; dx <= actionQRadius; dx++)
				for (int dy = -actionQRadius; dy <= actionQRadius; dy++) {
					int hx = centerX + dx;
					int hy = centerY + dy;

					if (hx >= 0 && hx < layerDescs.front()._width && hy >= 0 && hy < layerDescs.front()._height) {
						int hi = hx + hy * layerDescs.front()._width;

						{
							Connection c;

							c._weight = dist01(generator);
							c._index = hi;

							pvNode._firstHiddenConnections.push_back(c);

							dist2 += c._weight * c._weight;
						}

						// Secondary prediction connection
						{
							Connection c;

							c._weight = dist01(generator);
							c._index = hi;

							pvNode._firstHiddenConnections.push_back(c);

							dist2 += c._weight * c._weight;
						}
					}
				}

			pvNode._firstHiddenConnections.shrink_to_fit();

			float normMult = 1.0f / dist2;

			for (int ci = 0; ci < pvNode._firstHiddenConnections.size(); ci++)
				pvNode._firstHiddenConnections[ci]._weight *= normMult;

			_pvNodes.push_back(pvNode);

			break;
		}
		case _lv:
		{
			Node lvNode;

			lvNode._inputIndex = vi;
			lvNode._firstHiddenConnections.reserve(actionQSize);

			int centerX = std::round(visibleToHiddenWidth * vx);
			int centerY = std::round(visibleToHiddenHeight * vy);

			float dist2 = 0.0f;

			for (int dx = -actionQRadius; dx <= actionQRadius; dx++)
				for (int dy = -actionQRadius; dy <= actionQRadius; dy++) {
					int hx = centerX + dx;
					int hy = centerY + dy;

					if (hx >= 0 && hx < layerDescs.front()._width && hy >= 0 && hy < layerDescs.front()._height) {
						int hi = hx + hy * layerDescs.front()._width;

						{
							Connection c;

							c._weight = dist01(generator);
							c._index = hi;

							lvNode._firstHiddenConnections.push_back(c);

							dist2 += c._weight * c._weight;
						}

						// Secondary prediction connection
						{
							Connection c;

							c._weight = dist01(generator);
							c._index = hi;

							lvNode._firstHiddenConnections.push_back(c);

							dist2 += c._weight * c._weight;
						}
					}
				}

			lvNode._firstHiddenConnections.shrink_to_fit();

			float normMult = 1.0f / dist2;

			for (int ci = 0; ci < lvNode._firstHiddenConnections.size(); ci++)
				lvNode._firstHiddenConnections[ci]._weight *= normMult;

			_lvNodes.push_back(lvNode);

			break;
		}
		}
	}

	_htsl.createRandom(inputWidth, inputHeight, layerDescs, generator);
}

void HTSLRL::update(float reward, std::mt19937 &generator) {
	for (int ni = 0; ni < _pvNodes.size(); ni++)
		_htsl.setInput(_pvNodes[ni]._inputIndex, reward);

	float pvError = reward - _expectedReward;

	bool pvFilter = reward > 0.5f && _expectedReward > 0.5f;

	float lvError;

	if (pvFilter) {
		for (int ni = 0; ni < _lvNodes.size(); ni++)
			_htsl.setInput(_lvNodes[ni]._inputIndex, reward);

		lvError = reward - _expectedSecondary;
	}
	else {
		for (int ni = 0; ni < _lvNodes.size(); ni++)
			_htsl.setInput(_lvNodes[ni]._inputIndex, 0.0f);

		lvError = -_expectedSecondary;
	}

	float error = pvFilter ? pvError : lvError;

	if (error > 0.0f) {
		for (int ni = 0; ni < _actionNodes.size(); ni++) {
			_htsl.setInput(_actionNodes[ni]._inputIndex, _actionNodes[ni]._output);
		}
	}
	else {
		for (int ni = 0; ni < _actionNodes.size(); ni++) {
			_htsl.setInput(_actionNodes[ni]._inputIndex, _actionNodes[ni]._state);
		}
	}

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::normal_distribution<float> perturbationDist(0.0f, _actionPerturbationStdDev);

	_htsl.update();
	_htsl.learnRSC();
	_htsl.learnPrediction();

	// Collect expected reward
	float pvSum = 0.0f;

	for (int ni = 0; ni < _pvNodes.size(); ni++) {
		pvSum += _htsl.getPrediction(_pvNodes[ni]._inputIndex);
	}

	float expectedReward = pvSum / _pvNodes.size();

	float lvSum = 0.0f;

	for (int ni = 0; ni < _lvNodes.size(); ni++) {
		lvSum += _htsl.getPrediction(_lvNodes[ni]._inputIndex);
	}

	float expectedSecondary = lvSum / _lvNodes.size();

	// Derive new actions
	for (int ni = 0; ni < _actionNodes.size(); ni++) {
		_actionNodes[ni]._state = _htsl.getPrediction(_actionNodes[ni]._inputIndex);

		if (dist01(generator) < _actionRandomizeChance)
			_actionNodes[ni]._output = dist01(generator);
		else
			_actionNodes[ni]._output = std::min(1.0f, std::max(0.0f, std::min(1.0f, std::max(0.0f, _actionNodes[ni]._state)) + perturbationDist(generator)));
	}

	_expectedReward = expectedReward;
	_expectedSecondary = expectedSecondary;

	std::cout << reward << " " << _expectedReward << " " << _expectedSecondary << " " << error << " " << (pvFilter ? "F" : "N") << std::endl;

	_htsl.stepEnd();
}