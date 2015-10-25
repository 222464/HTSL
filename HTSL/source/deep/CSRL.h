#pragma once

#include "SDRRL.h"

#include <array>

namespace deep {
	class CSRL {
	public:
		struct Column {
			SDRRL _sou; // Outputs: states for next layer

			std::vector<float> _prevStates;

			std::vector<int> _ffIndices;
			std::vector<int> _lIndices;
			std::vector<int> _fbIndices;

			Column()
			{}
		};

		struct LayerDesc {
			int _width, _height;
			int _cellsPerColumn;
			int _ffStateActions, _lStateActions, _fbStateActions;
			int _recurrentActions;

			int _ffRadius, _lRadius, _fbRadius;

			float _ffAlpha, _inhibAlpha, _biasAlpha;

			float _qAlpha;
			float _actionAlpha;

			float _actionDeriveIterations;
			float _actionDeriveAlpha;
			float _actionDeriveStdDev;

			float _averageSurpriseDecay;
			float _surpriseLearnFactor;

			float _expPert;
			float _expBreak;

			float _gamma, _lambdaGamma;

			float _cellSparsity;

			LayerDesc()
				: _width(16), _height(16),
				_cellsPerColumn(16),
				_ffStateActions(1), _lStateActions(1), _fbStateActions(1),
				_recurrentActions(2),
				_ffRadius(2), _lRadius(2), _fbRadius(2),
				_ffAlpha(0.01f), _inhibAlpha(0.05f), _biasAlpha(0.005f),
				_qAlpha(0.01f),
				_actionAlpha(0.05f),
				_actionDeriveIterations(16), _actionDeriveAlpha(0.04f), _actionDeriveStdDev(0.01f),
				_averageSurpriseDecay(0.01f), _surpriseLearnFactor(2.0f),
				_expPert(0.02f),
				_expBreak(0.007f),
				_gamma(0.993f), _lambdaGamma(0.985f),
				_cellSparsity(0.125f)
			{}
		};

		struct Layer {
			std::vector<Column> _columns;
		};

	private:
		std::vector<LayerDesc> _layerDescs;
		std::vector<Layer> _layers;

		int _inputsPerState;

	public:
		void createRandom(int inputsPerState, const std::vector<LayerDesc> &layerDescs, float initMinWeight, float initMaxWeight, float initMinInhibition, float initMaxInhibition, float initThreshold, std::mt19937 &generator);

		void setState(int index, int input, float value) {
			_layers.front()._columns[index]._sou.setState(input, value);
		}

		void setState(int x, int y, int input, float value) {
			setState(x + _layerDescs.front()._width * y, input, value);
		}

		float getAction(int l, int index, int state) const {
			return _layers[l]._columns[index]._sou.getAction(state);
		}

		float getAction(int l, int x, int y, int state) const {
			return getAction(l, x + _layerDescs.front()._width * y, state);
		}

		void simStep(int subIter, float reward, std::mt19937 &generator);

		const std::vector<LayerDesc> &getLayerDescs() const {
			return _layerDescs;
		}

		const std::vector<Layer> &getLayers() const {
			return _layers;
		}
	};
}