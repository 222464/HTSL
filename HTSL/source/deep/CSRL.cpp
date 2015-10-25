#include "CSRL.h"

using namespace deep;

void CSRL::createRandom(int inputsPerState, const std::vector<LayerDesc> &layerDescs, float initMinWeight, float initMaxWeight, float initMinInhibition, float initMaxInhibition, float initThreshold, std::mt19937 &generator) {
	_inputsPerState = inputsPerState;
	
	_layerDescs = layerDescs;

	_layers.resize(_layerDescs.size());

	for (int l = 0; l < _layers.size(); l++) {
		LayerDesc &desc = _layerDescs[l];
		Layer &layer = _layers[l];

		int ffDiam = desc._ffRadius * 2 + 1;
		int lDiam = desc._lRadius * 2 + 1;
		int fbDiam = desc._fbRadius * 2 + 1;

		int ffSize;
		
		if (l > 0)
			ffSize = ffDiam * ffDiam;
		else
			ffSize = 1; // Input layer

		int lSize = lDiam * lDiam;

		int fbSize;

		if (l < _layers.size() - 1)
			fbSize = fbDiam * fbDiam;
		else
			fbSize = 0; // No next layer

		int totalSize = ffSize + lSize + fbSize;

		layer._columns.resize(desc._width * desc._height);

		for (int c = 0; c < layer._columns.size(); c++) {
			Column &col = layer._columns[c];

			int cx = c % desc._width;
			int cy = c / desc._width;

			col._prevStates.assign(desc._ffStateActions + desc._lStateActions + desc._fbStateActions, 0.0f);

			int inputSize = 0;

			if (l > 0) {
				LayerDesc &prevDesc = _layerDescs[l - 1];
				Layer &prevLayer = _layers[l - 1];

				int ffCenterX = std::round(cx * static_cast<float>(prevDesc._width) / static_cast<float>(desc._width));
				int ffCenterY = std::round(cy * static_cast<float>(prevDesc._height) / static_cast<float>(desc._height));

				// FF
				for (int dx = -desc._ffRadius; dx <= desc._ffRadius; dx++)
					for (int dy = -desc._ffRadius; dy <= desc._ffRadius; dy++) {
						int ox = ffCenterX + dx;
						int oy = ffCenterY + dy;

						if (ox >= 0 && ox < prevDesc._width && oy >= 0 && oy < prevDesc._height) {
							int oc = ox + oy * prevDesc._width;

							col._ffIndices.push_back(oc);

							inputSize += prevDesc._ffStateActions;
						}
					}
			}
			else
				inputSize += _inputsPerState; // For input layer

			{
				// Lateral
				for (int dx = -desc._lRadius; dx <= desc._lRadius; dx++)
					for (int dy = -desc._lRadius; dy <= desc._lRadius; dy++) {
						int ox = cx + dx;
						int oy = cy + dy;

						if (ox >= 0 && ox < desc._width && oy >= 0 && oy < desc._height) {
							int oc = ox + oy * desc._width;

							col._lIndices.push_back(oc);

							inputSize += desc._lStateActions;
						}
					}
			}

			if (l < _layers.size() - 1) {
				LayerDesc &nextDesc = _layerDescs[l + 1];
				Layer &nextLayer = _layers[l + 1];

				int fbCenterX = std::round(cx * static_cast<float>(nextDesc._width) / static_cast<float>(desc._width));
				int fbCenterY = std::round(cy * static_cast<float>(nextDesc._height) / static_cast<float>(desc._height));

				// FF
				for (int dx = -desc._fbRadius; dx <= desc._fbRadius; dx++)
					for (int dy = -desc._fbRadius; dy <= desc._fbRadius; dy++) {
						int ox = fbCenterX + dx;
						int oy = fbCenterY + dy;

						if (ox >= 0 && ox < nextDesc._width && oy >= 0 && oy < nextDesc._height) {
							int oc = ox + oy * nextDesc._width;

							col._fbIndices.push_back(oc);

							inputSize += nextDesc._fbStateActions;
						}
					}
			}

			// Recurrent actions
			inputSize += desc._recurrentActions;

			col._sou.createRandom(inputSize, desc._ffStateActions + desc._lStateActions + desc._fbStateActions + desc._recurrentActions, desc._cellsPerColumn, initMinWeight, initMaxWeight, initMinInhibition, initMaxInhibition, initThreshold, generator);
		}
	}
}

void CSRL::simStep(int subIter, float reward, std::mt19937 &generator) {
	for (int iter = 0; iter < subIter; iter++) {
		for (int l = 0; l < _layers.size(); l++) {
			LayerDesc &desc = _layerDescs[l];
			Layer &layer = _layers[l];

			int ffDiam = desc._ffRadius * 2 + 1;
			int lDiam = desc._lRadius * 2 + 1;
			int fbDiam = desc._fbRadius * 2 + 1;

			int ffSize = ffDiam * ffDiam;
			int lSize = lDiam * lDiam;
			int fbSize = fbDiam * fbDiam;

			int totalSize = ffSize + lSize + fbSize;

			for (int c = 0; c < layer._columns.size(); c++) {
				Column &col = layer._columns[c];

				int cx = c % desc._width;
				int cy = c / desc._width;

				int index = 0;

				if (l > 0) {
					LayerDesc &prevDesc = _layerDescs[l - 1];
					Layer &prevLayer = _layers[l - 1];

					for (int i = 0; i < col._ffIndices.size(); i++) {
						for (int j = 0; j < prevDesc._ffStateActions; j++)
							col._sou.setState(index++, prevLayer._columns[col._ffIndices[i]]._prevStates[j]);
					}
				}
				else
					index += _inputsPerState; // For input layer

				{
					for (int i = 0; i < col._lIndices.size(); i++)
						for (int j = 0; j < desc._lStateActions; j++)
							col._sou.setState(index++, layer._columns[col._lIndices[i]]._prevStates[j + desc._ffStateActions]);
				}

				if (l < _layers.size() - 1) {
					LayerDesc &nextDesc = _layerDescs[l + 1];
					Layer &nextLayer = _layers[l + 1];

					for (int i = 0; i < col._fbIndices.size(); i++)
						for (int j = 0; j < nextDesc._fbStateActions; j++)
							col._sou.setState(index++, nextLayer._columns[col._fbIndices[i]]._prevStates[j + nextDesc._ffStateActions + nextDesc._lStateActions]);
				}

				// Recurrent actions
				for (int r = 0; r < desc._recurrentActions; r++)
					col._sou.setState(index++, col._sou.getAction(desc._ffStateActions + desc._lStateActions + desc._fbStateActions + r));

				// Column update
				col._sou.simStep(reward, desc._cellSparsity, desc._gamma, desc._ffAlpha, desc._inhibAlpha, desc._biasAlpha, desc._qAlpha, desc._actionAlpha, desc._actionDeriveIterations, desc._actionDeriveAlpha, desc._lambdaGamma, desc._expPert, desc._expBreak, desc._averageSurpriseDecay, desc._surpriseLearnFactor, generator);
			}
		}

		// Buffer update
		for (int l = 0; l < _layers.size(); l++) {
			LayerDesc &desc = _layerDescs[l];
			Layer &layer = _layers[l];

			for (int c = 0; c < layer._columns.size(); c++) {
				Column &col = layer._columns[c];

				for (int s = 0; s < col._prevStates.size(); s++)
					col._prevStates[s] = col._sou.getAction(s);
			}
		}
	}
}