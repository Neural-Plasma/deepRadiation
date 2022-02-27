# deepRadiation

A Deep Neural Network-based radiation pressure prediction using TensorFlow.

<!-- ![deep diffusion](deep_diff.gif) -->

## Contributors
- [Rinku Mishra](https://github.com/rinku-mishra), IPR, India.
- [Sayan Adhikari](https://github.com/sayanadhikari), UiO, Norway. [@sayanadhikari](https://twitter.com/sayanadhikari)
- [Rupak Mukherjee](https://github.com/RupakMukherjee), PPPL, USA.

## Installation
### Prerequisites
1. [python3 or higher](https://www.python.org/download/releases/3.0/)
2. [git](https://git-scm.com/)
3. [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

### Procedure
#### Using Anaconda/Miniconda
First make a clone of the master branch using the following command
```shell
git clone https://github.com/Neural-Plasma/deepRadiation.git
```
Then enter inside the *deepDiffusion* directory
```shell
cd deepRadiation
```
Now create a conda environment using the given *environment.yml* file
```shell
conda env create -f environment.yml
```
Activate the conda environment
```shell
conda activate deepRadiation
```
## Usage

Run the code using following command

#### To train the model
```
python deepDiff --train
```
#### To test the model
```
python deepDiff --test
```
## Parameter Setup
Edit the _input.ini_ and run the code again. The basic structure of _input.ini_ is provided below,

```ini
;
; @file		input.ini
; @brief	deepDiffusion inputfile.
; @author	Sayan Adhikari <sayan.adhikari@fys.uio.no>
;         Rupak Mukherjee <rupakm@princeton.edu>


[grid]
# box size, mm
w = 10.
h = 10.
# intervals in x-, y- directions, mm
dx = 0.1
dy = 0.1

[par]
# Thermal diffusivity of steel, mm2.s-1
D = 1.

[time]
# Number of timesteps
nsteps = 101
# time step to get data from dnn
dnn_start = 50

[dnn]
# number of neurons
nn = 100
epochs = 500
patience = 50
batch_size=32
nlayer = 6

[figures]
plot_fig = True
use_latex = True
add_labels = True

[diagnostics]
dumpData = True
```

## Contributing
We welcome contributions to this project.

1. Fork it.
2. Create your feature branch (```git checkout -b my-new-feature```).
3. Commit your changes (```git commit -am 'Add some feature'```).
4. Push to the branch (```git push origin my-new-feature```).
5. Create new Pull Request.

## License
Released under the [MIT license](LICENSE).
