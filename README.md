# deepRadiation
[![DOI](https://zenodo.org/badge/386719346.svg)](https://zenodo.org/badge/latestdoi/386719346)

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

```
python deepRad -l -hp -tp
```
To know about the arguments
```
python deepRad -h
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
