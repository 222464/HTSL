#include <Settings.h>

#if SUBPROGRAM_EXECUTE == SOUND_LEARNING

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
#include <deep/AutoEncoder.h>

#include <complex>
#include <valarray>

const double pi = 3.14159265359;

typedef std::complex<double> Complex;
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
		Complex t = std::polar(1.0, -2 * pi * k / N) * odd[k];
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
const int aeFeatures = 100;
const double sampleScalar = 1.0 / std::pow(2.0, 15.0);
const double sampleScalarInv = 1.0 / sampleScalar;
const int reconStride = 512;
const double sampleCurvePower = 1.0;
const double sampleCurvePowerInv = 1.0 / sampleCurvePower;
const int trainStride = 512;

double compress(double x) {
	return x * sampleScalar;// (x > 0 ? 1 : -1) * std::pow(std::abs(x) * sampleScalar, sampleCurvePower);
}

double decompress(double x) {
	return x * sampleScalarInv;// (x > 0 ? 1 : -1) * std::pow(std::abs(x), sampleCurvePowerInv) * sampleScalarInv;
}

int main() {
	std::mt19937 generator(time(nullptr));

	sf::SoundBuffer buffer;

	buffer.loadFromFile("testSound.wav");

	sc::ISparseCoder ae;

	ae.createRandom(aeSamplesSize, aeFeatures, 0.5, generator);

	std::cout << "Training low-level feature extractor..." << std::endl;

	std::uniform_int_distribution<int> soundDist(0, buffer.getSampleCount() - aeSamplesSize - 1);

	int featureIter = 10000;

	int iIter = 32;
	double iStepSize = 0.1;
	double iLambda = 0.5;
	double iEpsilon = 0.001;

	for (int t = 0; t < featureIter; t++) {
		int i = soundDist(generator);

		for (int s = 0; s < aeSamplesSize; s++)
			ae.setVisibleInput(s, compress(buffer.getSamples()[i + s]));

		ae.activate(iIter, iStepSize, iLambda, iEpsilon);

		ae.reconstruct();

		ae.learn(0.01);

		if (t % 1000 == 0) {
			std::cout << static_cast<int>(static_cast<float>(t) / featureIter * 100.0f) << "%" << std::endl;
		}
	}

	std::cout << "Learned features." << std::endl;
	std::cout << "Testing sound reconstruction..." << std::endl;

	std::vector<double> reconSamplesf(buffer.getSampleCount(), 0.0);
	std::vector<double> convCounts(buffer.getSampleCount(), 0.0);

	double sampleRangeInv = 1.0 / (0.5 * aeSamplesSize);

	for (int i = 0; i < buffer.getSampleCount() - aeSamplesSize; i += reconStride) {
		for (int s = 0; s < aeSamplesSize; s++)
			ae.setVisibleInput(s, compress(buffer.getSamples()[i + s]));

		ae.activate(iIter, iStepSize, iLambda, iEpsilon);

		ae.reconstruct();

		for (int s = 0; s < aeSamplesSize; s++) {
			double intensity = 1.0;// std::max(0.0, 1.0 - std::abs(0.5 * aeSamplesSize - s) * sampleRangeInv);
			reconSamplesf[i + s] += ae.getVisibleRecon(s) * intensity;
			convCounts[i + s] += intensity;
		}
	}

	for (int i = 0; i < reconSamplesf.size(); i++)
		if (convCounts[i] != 0)
			reconSamplesf[i] /= static_cast<double>(convCounts[i]);

	std::vector<sf::Int16> reconSamples(buffer.getSampleCount());

	// Normalize
	double mean = 0.0;

	for (int i = 0; i < reconSamplesf.size(); i++)
		mean += reconSamplesf[i];

	mean /= reconSamplesf.size();

	double mag2 = 0.0;

	for (int i = 0; i < reconSamplesf.size(); i++) {
		reconSamplesf[i] -= mean;

		mag2 += reconSamplesf[i] * reconSamplesf[i];
	}

	double magInv = 100.0 / std::sqrt(mag2);

	for (int i = 0; i < reconSamplesf.size(); i++)
		reconSamplesf[i] *= magInv;

	for (int i = 0; i < reconSamples.size(); i++)
		reconSamples[i] = static_cast<sf::Int16>(decompress(reconSamplesf[i]));

	sf::SoundBuffer recon;

	recon.loadFromSamples(reconSamples.data(), reconSamples.size(), 1, buffer.getSampleRate());

	recon.saveToFile("reconSound.wav");

	std::cout << "Sound reconstructed." << std::endl;
	std::cout << "Training on sound..." << std::endl;

	sc::HTSL ht;

	std::vector<sc::HTSL::LayerDesc> layerDescs(3);

	layerDescs[0]._width = 16;
	layerDescs[0]._height = 16;

	layerDescs[1]._width = 12;
	layerDescs[1]._height = 12;

	layerDescs[2]._width = 8;
	layerDescs[2]._height = 8;

	ht.createRandom(10, 10, layerDescs, generator);

	int featuresCount = static_cast<int>(std::floor(buffer.getSampleCount() / static_cast<float>(trainStride)));

	for (int t = 0; t < 5; t++) {
		for (int s = 0; s < featuresCount; s++) {
			// Extract features
			int start = s * trainStride;

			for (int i = 0; i < trainStride; i++) {
				int si = start + i;

				if (si < buffer.getSampleCount())
					ae.setVisibleInput(i, compress(buffer.getSamples()[si]));
				else
					ae.setVisibleInput(i, 0.0);
			}

			ae.activate(iIter, iStepSize, iLambda, iEpsilon);

			// Run on extracted features
			for (int f = 0; f < aeFeatures; f++)
				ht.setInput(f, ae.getHiddenActivation(f));

			ht.update();

			ht.learn();

			ht.stepEnd();
		}

		std::cout << "Pass " << t << std::endl;
	}

	std::cout << "Generating extra..." << std::endl;

	// Extend song
	int extraFeatures = 2000;

	std::vector<double> extraSamplesf(extraFeatures * trainStride, 0.0);
	std::vector<sf::Uint16> extraSamplesSums(extraFeatures * trainStride, 0);

	for (int s = 0; s < extraFeatures; s++) {
		if (s < featuresCount) {
			// Extract features
			std::vector<float> features(aeFeatures);

			int start = s * trainStride;

			for (int i = 0; i < trainStride; i++) {
				int si = start + i;

				if (si < buffer.getSampleCount())
					ae.setVisibleInput(i, compress(buffer.getSamples()[start + i]));
				else
					ae.setVisibleInput(i, 0.0);
			}

			ae.activate(iIter, iStepSize, iLambda, iEpsilon);

			for (int f = 0; f < aeFeatures; f++)
				ht.setInput(f, ae.getHiddenActivation(f));
		}
		else {
			// Run on extracted features
			for (int f = 0; f < aeFeatures; f++)
				ht.setInput(f, ht.getPrediction(f));
		}

		ht.update();

		ht.stepEnd();

		std::vector<double> pred(aeFeatures);

		for (int f = 0; f < aeFeatures; f++)
			pred[f] = ht.getPrediction(f);

		std::vector<double> reconstruction(trainStride);

		ae.reconstruct(pred, reconstruction);

		int start = s * trainStride;

		for (int i = 0; i < trainStride; i++) {
			extraSamplesf[start + i] += reconstruction[i];
			extraSamplesSums[start + i]++;
		}
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