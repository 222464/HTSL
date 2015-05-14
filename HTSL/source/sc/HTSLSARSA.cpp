#include "HTSLSARSA.h"

#include <algorithm>

#include <iostream>

using namespace sc;

void HTSLSARSA::createRandom(int inputWidth, int inputHeight, int actionQRadius, const std::vector<InputType> &inputTypes, const std::vector<HTSL::LayerDesc> &layerDescs, std::mt19937 &generator) {
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

			float normMult = 1.0f / std::sqrt(dist2);

			for (int ci = 0; ci < actionNode._firstHiddenConnections.size(); ci++)
				actionNode._firstHiddenConnections[ci]._weight *= normMult;

			_actionNodes.push_back(actionNode);

			break;
		}
		case _q:
		{
			Node qNode;

			qNode._inputIndex = vi;
			qNode._firstHiddenConnections.reserve(actionQSize);

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

							qNode._firstHiddenConnections.push_back(c);

							dist2 += c._weight * c._weight;
						}

						// Secondary prediction connection
						{
							Connection c;

							c._weight = dist01(generator);
							c._index = hi;

							qNode._firstHiddenConnections.push_back(c);

							dist2 += c._weight * c._weight;
						}
					}
				}

			qNode._firstHiddenConnections.shrink_to_fit();

			float normMult = 1.0f / std::sqrt(dist2);

			for (int ci = 0; ci < qNode._firstHiddenConnections.size(); ci++)
				qNode._firstHiddenConnections[ci]._weight *= normMult;

			_qNodes.push_back(qNode);

			break;
		}
		}
	}

	_htsl.createRandom(inputWidth, inputHeight, layerDescs, generator);
}

void HTSLSARSA::update(float reward, std::mt19937 &generator) {
	for (int ni = 0; ni < _qNodes.size(); ni++)
		_htsl.setInput(_qNodes[ni]._inputIndex, _prevNewQ);

	for (int ni = 0; ni < _actionNodes.size(); ni++)
		_htsl.setInput(_actionNodes[ni]._inputIndex, _actionNodes[ni]._output);

	_htsl.update();
	_htsl.learn();

	// Collect Q
	float qSum = 0.0f;

	for (int ni = 0; ni < _qNodes.size(); ni++) {
		for (int ci = 0; ci < _qNodes[ni]._firstHiddenConnections.size(); ci++) {
			if (ci % 2 == 0)
				qSum += _qNodes[ni]._firstHiddenConnections[ci]._weight * _htsl.getLayers().front()._rsc.getHiddenBit(_qNodes[ni]._firstHiddenConnections[ci]._index);
			else
				qSum += _qNodes[ni]._firstHiddenConnections[ci]._weight * std::min(1.0f, std::max(0.0f, _htsl.getLayers().front()._predictionNodes[_qNodes[ni]._firstHiddenConnections[ci]._index]._state));
		}
	}

	float nextQ = qSum;

	float newQ = reward + _qGamma * nextQ;

	float tdError = newQ - _prevValue;

	std::cout << nextQ << " " << newQ << " " << tdError << " " << _actionNodes[0]._state << std::endl;

	for (int ni = 0; ni < _qNodes.size(); ni++) {
		float alphaError = _qAlpha * tdError;

		for (int ci = 0; ci < _qNodes[ni]._firstHiddenConnections.size(); ci++) {
			_qNodes[ni]._firstHiddenConnections[ci]._weight += alphaError * _qNodes[ni]._firstHiddenConnections[ci]._trace;

			if (ci % 2 == 0)
				_qNodes[ni]._firstHiddenConnections[ci]._trace = std::max((1.0f - _qTraceDecay) * _qNodes[ni]._firstHiddenConnections[ci]._trace, _htsl.getLayers().front()._rsc.getHiddenBit(_qNodes[ni]._firstHiddenConnections[ci]._index));
			else
				_qNodes[ni]._firstHiddenConnections[ci]._trace = std::max((1.0f - _qTraceDecay) * _qNodes[ni]._firstHiddenConnections[ci]._trace, std::min(1.0f, std::max(0.0f, _htsl.getLayers().front()._predictionNodes[_qNodes[ni]._firstHiddenConnections[ci]._index]._state)));
		}
	}

	float actionLearn = tdError > 0.0f ? _actionAlpha : 0.0f;// std::max(0.0f, _actionAlpha * tdError);

	for (int ni = 0; ni < _actionNodes.size(); ni++) {
		for (int ci = 0; ci < _actionNodes[ni]._firstHiddenConnections.size(); ci++)
			_actionNodes[ni]._firstHiddenConnections[ci]._weight += actionLearn * _actionNodes[ni]._firstHiddenConnections[ci]._trace;
	}

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::normal_distribution<float> perturbationDist(0.0f, _actionPerturbationStdDev);

	// Derive new actions
	for (int ni = 0; ni < _actionNodes.size(); ni++) {
		float sum = 0.0f;

		for (int ci = 0; ci < _actionNodes[ni]._firstHiddenConnections.size(); ci++) {
			if (ci % 2 == 0)
				sum += _actionNodes[ni]._firstHiddenConnections[ci]._weight * _htsl.getLayers().front()._rsc.getHiddenBit(_actionNodes[ni]._firstHiddenConnections[ci]._index);
			else
				sum += _actionNodes[ni]._firstHiddenConnections[ci]._weight * std::min(1.0f, std::max(0.0f, _htsl.getLayers().front()._predictionNodes[_actionNodes[ni]._firstHiddenConnections[ci]._index]._state));
		}

		_actionNodes[ni]._state = sum;

		if (dist01(generator) < _actionRandomizeChance)
			_actionNodes[ni]._output = dist01(generator);
		else
			_actionNodes[ni]._output = std::min(1.0f, std::max(0.0f, std::min(1.0f, std::max(0.0f, _actionNodes[ni]._state)) + perturbationDist(generator)));
	}

	// Update traces
	for (int ni = 0; ni < _actionNodes.size(); ni++) {
		float delta = _actionNodes[ni]._output - _actionNodes[ni]._state;

		for (int ci = 0; ci < _actionNodes[ni]._firstHiddenConnections.size(); ci++) {
			if (ci % 2 == 0)
				_actionNodes[ni]._firstHiddenConnections[ci]._trace = (1.0f - _actionTraceDecay) * _actionNodes[ni]._firstHiddenConnections[ci]._trace + _actionBeta * std::exp(-std::abs(_actionNodes[ni]._firstHiddenConnections[ci]._trace) * _actionTraceTemperature) * delta * _htsl.getLayers().front()._rsc.getHiddenBit(_actionNodes[ni]._firstHiddenConnections[ci]._index);
			else
				_actionNodes[ni]._firstHiddenConnections[ci]._trace = (1.0f - _actionTraceDecay) * _actionNodes[ni]._firstHiddenConnections[ci]._trace + _actionBeta * std::exp(-std::abs(_actionNodes[ni]._firstHiddenConnections[ci]._trace) * _actionTraceTemperature) * delta * std::min(1.0f, std::max(0.0f, _htsl.getLayers().front()._predictionNodes[_actionNodes[ni]._firstHiddenConnections[ci]._index]._state));
		}
	}

	_prevValue = nextQ;
	_prevNewQ = newQ;

	_htsl.stepEnd();
}