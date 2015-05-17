#include "HTSLPVLV.h"

#include <algorithm>

#include <iostream>

using namespace sc;

void HTSLPVLV::createRandom(int inputWidth, int inputHeight, const std::vector<InputType> &inputTypes, const std::vector<HTSL::LayerDesc> &layerDescs, std::mt19937 &generator) {
	assert(inputTypes.size() == inputWidth * inputHeight);

	_inputTypes = inputTypes;

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

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
	
			_actionNodes.push_back(actionNode);

			break;
		}
		case _pv:
		{
			Node pvNode;

			pvNode._inputIndex = vi;
			
			_pvNodes.push_back(pvNode);

			break;
		}
		case _lve:
		{
			Node lvNode;

			lvNode._inputIndex = vi;

			_lveNodes.push_back(lvNode);

			break;
		}
		case _lvi:
		{
			Node lvNode;

			lvNode._inputIndex = vi;
			
			_lviNodes.push_back(lvNode);

			break;
		}
		}
	}

	_htsl.createRandom(inputWidth, inputHeight, layerDescs, generator);
}

void HTSLPVLV::update(float reward, std::mt19937 &generator) {
	float targetP = _expectedReward + _expectedAlpha * (reward - _expectedReward);

	for (int ni = 0; ni < _pvNodes.size(); ni++)
		_htsl.setInput(_pvNodes[ni]._inputIndex, targetP);

	float pvError = reward - _expectedReward;

	bool pvFilter = (reward < _thetaMin || _expectedReward < _thetaMin) || (reward > _thetaMax || _expectedReward > _thetaMax);

	float lvError;

	if (pvFilter) {
		float targetE = _expectedSecondaryE + _secondaryAlphaE * (reward - _expectedSecondaryE);
		float targetI = _expectedSecondaryI + _secondaryAlphaI * (reward - _expectedSecondaryI);

		for (int ni = 0; ni < _lveNodes.size(); ni++)
			_htsl.setInput(_lveNodes[ni]._inputIndex, targetE);

		for (int ni = 0; ni < _lviNodes.size(); ni++)
			_htsl.setInput(_lviNodes[ni]._inputIndex, targetI);
	}
	else {
		for (int ni = 0; ni < _lveNodes.size(); ni++)
			_htsl.setInput(_lveNodes[ni]._inputIndex, _expectedSecondaryE);

		for (int ni = 0; ni < _lviNodes.size(); ni++)
			_htsl.setInput(_lviNodes[ni]._inputIndex, _expectedSecondaryI);
	}

	lvError = _expectedSecondaryE - _expectedSecondaryI;

	float error = pvFilter ? pvError : lvError;// (lvError + pvError);

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

	_htsl.learn();

	// Collect expected reward
	float pvSum = 0.0f;

	for (int ni = 0; ni < _pvNodes.size(); ni++) {
		pvSum += _htsl.getPrediction(_pvNodes[ni]._inputIndex);
	}

	float expectedReward = pvSum / _pvNodes.size();

	float lveSum = 0.0f;

	for (int ni = 0; ni < _lveNodes.size(); ni++) {
		lveSum += _htsl.getPrediction(_lveNodes[ni]._inputIndex);
	}

	float expectedSecondaryE = lveSum / _lveNodes.size();

	float lviSum = 0.0f;

	for (int ni = 0; ni < _lviNodes.size(); ni++) {
		lviSum += _htsl.getPrediction(_lviNodes[ni]._inputIndex);
	}

	float expectedSecondaryI = lviSum / _lviNodes.size();

	// Derive new actions
	for (int ni = 0; ni < _actionNodes.size(); ni++) {
		_actionNodes[ni]._state = _htsl.getPrediction(_actionNodes[ni]._inputIndex);

		if (dist01(generator) < _actionRandomizeChance)
			_actionNodes[ni]._output = dist01(generator);
		else
			_actionNodes[ni]._output = std::min(1.0f, std::max(0.0f, std::min(1.0f, std::max(0.0f, _actionNodes[ni]._state)) + perturbationDist(generator)));
	}

	std::cout << reward << " " << _expectedReward << " " << _expectedSecondaryE << " " << _expectedSecondaryI << " " << error << " " << (pvFilter ? "F" : "N") << std::endl;

	_expectedReward = expectedReward;
	_expectedSecondaryE = expectedSecondaryE;
	_expectedSecondaryI = expectedSecondaryI;

	_htsl.stepEnd();
}