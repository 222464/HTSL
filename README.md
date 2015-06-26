HTSL
=======

HTSL is a derivative of HTM (hierarchical temporal memory). It extends HTM with continuous input/output, temporal pooling, full hierarchy support, and sensory-motor integration.

Overview
-----------

HTSL stands for hierarchical temporal sparse learner. It can be thought of as a echo state network with a self-optimizing hierarchical reservoir.
HTSL currently only runs on the CPU. For a GPU alternative, you may want to look at HEInetGPU.

This repository contains several demos:
- Slime volleyball (RL game)
- Image prediction (video frame prediction)
- Music note prediction/generation
- Generic sound prediction/generation
- Mountain car
- Pole balancing
- Character prediction

Algorithm Description
-----------

HTSL is a stack of sparse coder.

The hidden representation of the sparse coders is kept sparse with explicit local lateral inhibition. This means that the resulting sparse distributed representations (SDRs) are local in the sense that they preserve the local topology of the input. This also vastly reduces forgetting, and means that no stochastic sampling is necessary. It also has recurrent hidden to hidden node connections, so it can handle partial observability.

HTSL has a feature known as temporal pooling: It can group previous events into conceptual "bins", as a by-product of the way the spatial pooling interacts with the hidden to hidden recurrent connections. The temporal pooling also allows the system to function very similarly to an echo state network, where the reservoir optimizes itself (since the SDRs change rather slowly).

HTSL has an up pass and a down pass: In the up pass features are extracted, and in the down pass predictions are made by refining predictions at each layer with information from that layer. This is accomplished with error-driven feedback weights, which learn to predict the next SDR (next timestep). These predictions also influence the formation of recurrent hidden-hidden weights.

Install
-----------

HTSL uses the CMake build system, and SFML for visualization.

You will need to have a version of SFML available in order to use HTSL's demos, since these require visualization.

You can find SFML here: [http://www.sfml-dev.org/download.php](http://www.sfml-dev.org/download.php)

Usage
-----------

To select a demo, change the SUBPROGRAM_EXECUTE macro to the demo of your choice, and then compile.

For using HTSL itself, you simply need to instantiate a HTSL object, and then call its .createRandom function (this initializes a HTSL network with random weights).
You will need to pass a vector of layer descriptions, which describe each layer in the hierarchy.

```cpp
sc::HTSL htsl;

std::vector<sc::HTSL::LayerDesc> layerDescs(3);

layerDescs[0]._width = 16;
layerDescs[0]._height = 16;

layerDescs[1]._width = 12;
layerDescs[1]._height = 12;

layerDescs[2]._width = 8;
layerDescs[2]._height = 8;

htsl.createRandom(inputWidth, inputHeight, layerDescs, generator);
```

You can then set the inputs to the HTSL network like so:

```cpp
htsl.setInput(index, value);
```

or the 2D version, which computes the index for you from 2D coordinates:

```cpp
htsl.setInput(x, y, value);
```

Then you can step the network one simulation tick like so:

```cpp
htsl.update();

htsl.learn();

htsl.stepEnd();
```

Update computes the network states, learn updates the weights, and stepEnd marks the end of the simulation step (this is required, as it updates temporal information).

Predictions can then be read for the next timestep:

```cpp
htsl.getPrediction(index)
```

or the 2D version:

```cpp
htsl.getPrediction(x, y)
```

For more usage examples, please see the demos.

License
-----------

HTSL
Copyright (C) 2015 Eric Laukien

This software is provided 'as-is', without any express or implied
warranty.  In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
	claim that you wrote the original software. If you use this software
	in a product, an acknowledgment in the product documentation would be
	appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
	misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.

------------------------------------------------------------------------------

HTSL uses the following external libraries:

SFML

