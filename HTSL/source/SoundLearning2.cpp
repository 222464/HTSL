#include <Settings.h>

#if SUBPROGRAM_EXECUTE == SOUND_LEARNING_2

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <SFML/Audio.hpp>

#include <fstream>
#include <sstream>

#include <unordered_set>
#include <unordered_map>

#include <iostream>

#include <sc/HTSL.h>
#include <sc/SparseCoder.h>
#include <sc/ISparseCoder.h>
#include <sc/BISparseCoder.h>
#include <sdr/PredictiveRSDR.h>

#include <complex>
#include <valarray>

#define RECONSTRUCT_DIRECT 1

const float pi = 3.14159265359;

typedef std::complex<float> Complex;
typedef std::valarray<Complex> CArray;

// Cooley–Tukey FFT (in-place)
void fft(CArray& x) {
	const size_t N = x.size();

	if (N <= 1)
		return;

	// Divide
	CArray even = x[std::slice(0, N / 2, 2)];
	CArray  odd = x[std::slice(1, N / 2, 2)];

	// Conquer
	fft(even);
	fft(odd);

	// Combine
	for (size_t k = 0; k < N / 2; ++k) {
		Complex t = std::polar(1.0f, -2.0f * pi * k / N) * odd[k];
		x[k] = even[k] + t;
		x[k + N / 2] = even[k] - t;
	}
}

// Inverse fft (in-place)
void ifft(CArray& x) {
	// Conjugate the complex numbers
	x = x.apply(std::conj);

	// Forward fft
	fft(x);

	// Conjugate the complex numbers again
	x = x.apply(std::conj);

	// Scale the numbers
	x /= x.size();
}

const int aeSamplesSize = 1024;
const int aeFeatures = 1024;
const float sampleScalar = 1.0f / std::pow(2.0f, 15.0f);
const float sampleScalarInv = 1.0f / sampleScalar;
const int reconStride = 512;
const float sampleCurvePower = 1.0f;
const float sampleCurvePowerInv = 1.0f / sampleCurvePower;
const int trainStride = 512;

float compress(float x) {
	return x * sampleScalar;// (x > 0 ? 1 : -1) * std::pow(std::abs(x) * sampleScalar, sampleCurvePower);
}

float decompress(float x) {
	return x * sampleScalarInv;// (x > 0 ? 1 : -1) * std::pow(std::abs(x), sampleCurvePowerInv) * sampleScalarInv;
}

struct FeaturesPointer {
	int _bufferStart;

	std::vector<float> _features;

	float similarity(const std::vector<float> &otherFeatures) {
		float sum = 0.0f;

		for (int i = 0; i < _features.size(); i++) {
			sum += _features[i] * otherFeatures[i];
		}

		return sum;
	}
};

int main() {
	std::mt19937 generator(time(nullptr));

	sf::SoundBuffer buffer;

	buffer.loadFromFile("testSound.wav");

	sdr::RSDR ae;

	int dimV = std::ceil(std::sqrt(static_cast<float>(aeSamplesSize)));
	int dimH = std::ceil(std::sqrt(static_cast<float>(aeFeatures)));

	ae.createRandom(dimV, dimV, dimH, dimH, 24, 12, -1, -0.001f, 0.001f, 0.01f, 0.1f, 0.0f, generator);

	std::cout << "Training low-level feature extractor..." << std::endl;

	std::uniform_int_distribution<int> soundDist(0, buffer.getSampleCount() - aeSamplesSize - 1);

	int featureIter = 12000;

	float sparsity = 0.01f;

	std::valarray<Complex> fftArray(aeSamplesSize);

	for (int t = 0; t < featureIter; t++) {
		int i = soundDist(generator);

		for (int s = 0; s < aeSamplesSize; s++)
			fftArray[s] = Complex(compress(buffer.getSamples()[i + s]));

		//fft(fftArray);

		for (int s = 0; s < aeSamplesSize; s++)
			ae.setVisibleInput(s, fftArray[s].real());

		ae.activate(sparsity);

		ae.reconstruct();

		ae.learn(0.05f, 0.05f, 0.05f, 0.1f, sparsity);

		if (t % 100 == 0) {
			std::cout << (static_cast<float>(t) / featureIter * 100.0f) << "%" << std::endl;
		}
	}

	std::cout << "Learned features." << std::endl;
	std::cout << "Testing sound reconstruction..." << std::endl;

	std::vector<float> reconSamplesf(buffer.getSampleCount(), 0.0f);
	std::vector<float> convCounts(buffer.getSampleCount(), 0.0f);
	std::vector<FeaturesPointer> featuresPointers;

	featuresPointers.reserve((buffer.getSampleCount() - aeSamplesSize) / reconStride + 1);

	float sampleRangeInv = 1.0f / (0.5f * aeSamplesSize);

	for (int i = 0; i < buffer.getSampleCount() - aeSamplesSize; i += reconStride) {
		for (int s = 0; s < aeSamplesSize; s++)
			fftArray[s] = Complex(compress(buffer.getSamples()[i + s]));

		//fft(fftArray);

		for (int s = 0; s < aeSamplesSize; s++)
			ae.setVisibleInput(s, fftArray[s].real());

		ae.activate(sparsity);

		ae.reconstruct();

		for (int s = 0; s < aeSamplesSize; s++)
			fftArray[s] = Complex(compress(ae.getVisibleRecon(s)));

		//ifft(fftArray);

		// Create features pointer
		FeaturesPointer fp;

		fp._features.resize(ae.getNumHidden());

		for (int j = 0; j < ae.getNumHidden(); j++)
			fp._features[j] = ae.getHiddenActivation(j);

		fp._bufferStart = i;

		featuresPointers.push_back(fp);

		for (int s = 0; s < aeSamplesSize; s++) {
			float intensity = 1.0f;// std::max(0.0, 1.0 - std::abs(0.5 * aeSamplesSize - s) * sampleRangeInv);
			reconSamplesf[i + s] += fftArray[s].real() * intensity;
			convCounts[i + s] += intensity;
		}
	}

	for (int i = 0; i < reconSamplesf.size(); i++)
		if (convCounts[i] != 0)
			reconSamplesf[i] /= static_cast<float>(convCounts[i]);

	std::vector<sf::Int16> reconSamples(buffer.getSampleCount());

	// Normalize
	float mean = 0.0f;

	for (int i = 0; i < reconSamplesf.size(); i++)
		mean += reconSamplesf[i];

	mean /= reconSamplesf.size();

	float mag2 = 0.0f;

	for (int i = 0; i < reconSamplesf.size(); i++) {
		reconSamplesf[i] -= mean;

		mag2 += reconSamplesf[i] * reconSamplesf[i];
	}

	float magInv = 100.0f / std::sqrt(mag2);

	for (int i = 0; i < reconSamplesf.size(); i++)
		reconSamplesf[i] *= magInv;

	for (int i = 0; i < reconSamples.size(); i++)
		reconSamples[i] = static_cast<sf::Int16>(decompress(reconSamplesf[i]));

	sf::SoundBuffer recon;

	recon.loadFromSamples(reconSamples.data(), reconSamples.size(), 1, buffer.getSampleRate());

	recon.saveToFile("reconSound.wav");

	std::cout << "Sound reconstructed." << std::endl;
	std::cout << "Training on sound..." << std::endl;

	sdr::PredictiveRSDR ht;

	std::vector<sdr::PredictiveRSDR::LayerDesc> layerDescs(3);

	layerDescs[0]._width = 16;
	layerDescs[0]._height = 16;

	layerDescs[1]._width = 12;
	layerDescs[1]._height = 12;

	layerDescs[2]._width = 8;
	layerDescs[2]._height = 8;

	ht.createRandom(dimH, dimH, layerDescs, -0.001f, 0.001f, 0.01f, 0.1f, 0.0f, generator);

	int featuresCount = static_cast<int>(std::floor(buffer.getSampleCount() / static_cast<float>(trainStride)));

	for (int t = 0; t < 5; t++) {
		for (int s = 0; s < featuresCount; s++) {
			// Extract features
			int start = s * trainStride;

			for (int i = 0; i < aeSamplesSize; i++) {
				int si = start + i;

				if (si < buffer.getSampleCount())
					ae.setVisibleInput(i, compress(buffer.getSamples()[si]));
				else
					ae.setVisibleInput(i, 0.0f);
			}

			ae.activate(sparsity);

			// Run on extracted features
			for (int f = 0; f < aeFeatures; f++)
				ht.setInput(f, ae.getHiddenState(f));

			ht.simStep();
		}

		std::cout << "Pass " << t << std::endl;
	}

	std::cout << "Generating extra..." << std::endl;

	// Extend song
	int extraFeatures = 2000;

	std::vector<float> extraSamplesf((extraFeatures + 1) * trainStride, 0.0f);
	std::vector<float> extraSamplesSums((extraFeatures + 1) * trainStride, 0.0f);

	for (int s = 0; s < extraFeatures; s++) {
		if (s < featuresCount) {
			// Extract features
			std::vector<float> features(aeFeatures);

			int start = s * trainStride;

			for (int i = 0; i < aeSamplesSize; i++) {
				int si = start + i;

				if (si < buffer.getSampleCount())
					ae.setVisibleInput(i, compress(buffer.getSamples()[start + i]));
				else
					ae.setVisibleInput(i, 0.0f);
			}

			ae.activate(sparsity);

			for (int f = 0; f < aeFeatures; f++)
				ht.setInput(f, ae.getHiddenState(f));
		}
		else {
			// Run on extracted features
			for (int f = 0; f < aeFeatures; f++)
				ht.setInput(f, ht.getPrediction(f));
		}

		ht.simStep(false);

		std::vector<float> pred(aeFeatures);

		for (int f = 0; f < aeFeatures; f++)
			pred[f] = ht.getPrediction(f);

#if RECONSTRUCT_DIRECT
		std::vector<float> reconstruction(trainStride);

		ae.reconstructFeedForward(pred, reconstruction);

		int start = s * trainStride;

		for (int i = 0; i < aeSamplesSize; i++) {
			extraSamplesf[start + i] += reconstruction[i];
			extraSamplesSums[start + i]++;
		}
#else
		// Match features to closest in feature pointers
		float maxSim = -99999.0f;
		int maxFeatureIndex = 0;

		for (int f = 0; f < featuresPointers.size(); f++) {
			float sim = featuresPointers[f].similarity(pred);

			if (sim > maxSim) {
				maxSim = sim;

				maxFeatureIndex = f;
			}
		}

		int start = s * trainStride;

		for (int i = 0; i < aeSamplesSize; i++) {
			float intensity = maxSim * std::max(0.0f, 1.0f - std::abs(0.5f * aeSamplesSize - i) * sampleRangeInv);
			extraSamplesf[start + i] += intensity * compress(buffer.getSamples()[featuresPointers[maxFeatureIndex]._bufferStart + i]);
			extraSamplesSums[start + i] += intensity;
		}
#endif
	}

	std::vector<sf::Int16> extraSamples(extraSamplesf.size());

	for (int i = 0; i < extraSamplesf.size(); i++)
		if (extraSamplesSums[i] != 0)
			extraSamples[i] = static_cast<sf::Int16>(decompress(extraSamplesf[i] / extraSamplesSums[i]));

	sf::SoundBuffer extraBuffer;

	extraBuffer.loadFromSamples(extraSamples.data(), extraSamples.size(), 1, buffer.getSampleRate());

	std::cout << "Saving extra sound..." << std::endl;

	extraBuffer.saveToFile("extra.wav");

	std::cout << "Done." << std::endl;

	return 0;
}

#endif