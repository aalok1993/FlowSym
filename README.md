# Flow Symmetrization for Parameterized Constrained Diffeomorphisms

[[Paper](https://arxiv.org/abs/2312.06317)] [[Website]()]

## Installation

Install Anaconda using the instructions given [here](https://www.anaconda.com/docs/getting-started/anaconda/install). 

Use the following command to create a conda virtual environment with the required libraries.

`conda env create -f environment.yml`

Activate the conda environment

`conda activate flowsym`

## Usage (Escherization)

```
cd Escherization
python escherization.py --IH 04 --path images/rabbit --res Results/rabbit
```

In the above command, the `IH` flag represents the ID of the Isohedral class. The options are: 01, 02, 03, 04, 05, 06, 07, 21, 28. 
The `path` flag represents the path to the input image without the extension. THe image must be in PNG format where the input shape must be in black color and the background must be in white color. Sample images are contained in `Escherization/images`.
The `res` flag indicates the path where the results will be saved.
For exploring the additional flags use the command `python escherization.py --help`.

## Usage (Density Estimation)

```
cd DensityEstimation
python density_estimation.py --identification_space Torus --target_dist Checkerboard
```

In the above command, the `identification_space` flag represents the manifold on which density estimation is supposed to be performed. The options are: Torus, Sphere, Klein_Bottle, Projective_Space.
The `target_dist` flag represents the target probability distribution to be learned using density estimation. The options are: Checkerboard, FourGaussians, SixGaussians.
For exploring the additional flags use the command `python density_estimation.py --help`.
