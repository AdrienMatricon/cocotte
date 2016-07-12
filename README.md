# COCOTTE
COCOTTE (COnstrained Complexity Optimization Trough iTerative merging of Experts)
is an iterative algorithm for discovering and learning the smooth components of a
locally smooth function, which can be used to discover discrete, meaningful
forward-parameterized skills from a set of behaviour examples.

## Before starting
If you read about COCOTTE in a publication, you may be more interested in the version of COCOTTE
and its implementation described in this publication than in the latest one.
If so, please select the corresponding tag on the GitHub repository to look at the right version
of the code and of this README
(if you are not familiar with GitHub: click on `branch:master` at the top of this repository, then on the `Tags` tab,
and select the tag corresponding to the conference of journal you are looking for).

## License
COCOTTE's current implementation relies on four external libraries,
each under a specific license:
- Boost (http://www.boost.org), which we use for to dumping the learned models, is under the [Boost Software License][boost-license],
- OpenCV (http://opencv.org/), which has an implementation of Random Forests that we use for classification so that new points can be attributed to one of the learned models, is under a [BSD license][opencv-license],
- Soplex (http://soplex.zib.de/), which is a Linear Program solver that we use to compute infinity-norm minimization (formulated as a linear program), is under the [ZIB Academic License][soplex-license],
- GLPK  (https://www.gnu.org/software/glpk/), which is a Linear Program solver that we use when Soplex fails, is under the [GNU General Public License 3][glpk-license].
 
While the [Boost Software License][boost-license] and [BSD license][opencv-license] are not very restrictive when the code is only used (and not modified/redistributed), the other two licenses are:
- Software under the [GNU General Public License 3][glpk-license] cannot be used in proprietary programs. As such, COCOTTE is also free under [GNU General Public License 3][glpk-license].
- Software under the [ZIB Academic License][soplex-license] can only be used for academic purposes. Furthermore, every publication and presentation for which work based on this software or its output has been used must contain an appropriate citation and acknowledgment of the authors of this software. While COCOTTE's source code is free, you need Soplex to use it, which restricts its use to academic purposes in practice.

[boost-license]:http://www.boost.org/users/license.html
[opencv-license]:http://opencv.org/license.html
[soplex-license]:http://scip.zib.de/academic.txt
[glpk-license]:https://www.gnu.org/licenses/gpl.html

> I am currently looking for a replacement to the joint use of Soplex and GLPK.
> Not only are those libraries under very constraining licenses,
> they are also mostly optimized to solve one big linear program instead of repeatedly solving small ones as we do here,
> and they tend to return error codes ("not singular" for Soplex and "numerical difficulties" for GLPK)
> when an optimal solution could be found. As they do not seem to return error codes on the same problems,
> I am currently using mostly Soplex, and calling GLPK when it fails, which is ugly.

> If you happen to know a reliable Linear Program solving library that I could use instead of Soplex and GLPK,
> please send me a message (especially if it is under a non-restrictive license like BSD),
> for example by raising an issue
> (for those unfamiliar with GitHub: click on the `Issues` tab under the repository name then on `New Issue`).
> Please check that you are reading this on the latest version of the README and that I have not found one yet, though.

## Installing COCOTTE
#### Getting the files
Clone the repository, and if necessary switch to the version of COCOTTE you want to be working with using
```sh
# example: git checkout -b master 2016-icdl-epirob
$ git checkout -b [branchname] [tagname]
```

#### Boost
This code was developed under Boost 1.54 and uses `boost::serialization` to dump models.
It also uses `boost::shared_ptr` because `std::shared_ptr` is not compatible with `boost::serialization`.
```sh
$ sudo apt-get install libboost-all-dev
```
#### OpenCV
This code was developed under OpenCV 2.4.8 and uses `cv::RandomTrees` for classification.
```sh
$ sudo apt-get install libopencv-dev 
```
#### Soplex
This code wad developed under Soplex 2.2.0.

Create a folder `thirdparty` in the repository's main directory
(which we will assume is called `maindirectory` here, but is probably called `cocotte` if you did not rename it).
Download Soplex's source code archive from http://soplex.zib.de/#download, extract the folder it contains into `maindirectory/thirdparty`, and rename it `soplex`.
Now to compile the library:
```sh
# In maindirectory/thirdparty/soplex
$ make COMP=gnu OPT=opt ZLIB=false GMP=false
```
There should now be a file `maindirectory/thirdparty/soplex/lib/libsoplex.a`

#### GLPK
This code was developed under GLPK 4.52 and uses it when Soplex fails.
```sh
$ sudo apt-get install libglpk-dev
```

#### COCOTTE
Compiling COCOTTE requires `cmake`:
```sh
$ sudo apt-get install cmake
```
To compile COCOTTE, go into the main directory and:
```sh
# In COCOTTE's main directory
$ mkdir bin
$ cd bin
$ cmake ..
$ make
```
There should now be an executable file `maindirectory/bin/cocotte`, and two library files
`maindirectory/lib/libdatasources.a` and `maindirectory/lib/libcocotte.a`.

## Using COCOTTE as an executable
The executable generated from the code on this repository can be called in 3 ways:
- `cocotte learn` discovers models from a training dataset,
- `cocotte test` uses this file to predict outputs from a test dataset,
- `cocotte evaluate` produces an evaluation of one or more models over a test dataset

#### cocotte learn
Here is an example of `cocotte learn` call:
```sh
# cocotte learn <training data file> <data structure file> <nb points for training>
# <output file for models> [other output files for models]
$ cocotte learn data.csv structure.txt 300 models 0.txt models 1.txt models 2.txt
```
- `data.csv` is a CSV file containing the training data,
- `structure.txt` is a file containing information about the structure of the data (see below),
- `300` is the number of training points (that is to say, the number of lines read in `data.csv`),
- `models 0.txt models 1.txt models 2.txt` are files in which models will be dumped.
When there is only one file, all training points are processed by COCOTTE at once before dumping the learned models.
When there is more than one file, training points are divided into batches of roughly equal size and given
batch by batch to COCOTTE, so that intermediate models can be learned and dumped.

> As of now, the incremental version of COCOTTE is still buggy
> (it does not always return the same final models depending on the number of intermediate models),
> so we recommend only using one file.

> Please note that this implementation is memory based (which should be useful for future works):
> training datapoints are stored internally, and are dumped with the models.

#### Data structure file
The goal of COCOTTE is to learn the local smooth components of a function `t = F(x)`.

The role of the data structure file is to indicate which columns of the data are dimensions of `x` and which columns are dimensions of `t`.
Here is an example of structure file:
```
x x_prec
y 0.1
q0
q1 q1_prec
q2
z z_prec
q3
q4

x y z
```
- at the beginning of the file, each line contains the name of a column (indicating that it is a dimensions of either
`x` or `t`), possibly followed by a space and the precision on the values in this column
(either a constant value or the name of another column containing the precision for each datapoint),
- a blank line separates the two parts of the file,
- the last line contains the list of the dimensions of `t`, separated by spaces
(the other declared columns therefore being dimensions of `x`).
Note that those dimensions must have an associated precision.
- if similar lines are added after the last one, COCOTTE is run independently on each `t` defined by one such line.
Please note that any given column of the data can only exclusively be  in `x` or in one `t`.

#### cocotte test
Here is an example of `cocotte test` call:
```sh
# cocotte test <models data file> <test data file> <data structure file> <output file> [nb points to predict=10000]
$ cocotte test models.txt test.csv structure.txt output.csv 10000
```
- `models.txt` is a file in which models were dumped by `cocotte learn`,
- `test.csv` is a CSV file containing the test data,
- `structure.txt` is a file containing information about the structure of the data (see above),
- `output.csv` is a CSV file in which to dump `x` as well as the real and predicted `t` values for each test point,
- `10000` is the number of test points (that is to say, the number of lines read in `test.csv`).

#### cocotte evaluate
Here is an example of `cocotte evaluate` call:
```sh
# cocotte evaluate <input data file> <data structure file> <nb points for testing> <outputFile>
# <models data file> [other models data file]
$ cocotte evaluate test.csv structure.txt 10000 evaluation.csv models0.txt models1.txt models2.txt
```
- `test.csv` is a CSV file containing the test data,
- `structure.txt` is a file containing information about the structure of the data (see above),
- `10000` is the number of test points (that is to say, the number of lines read in `test.csv`),
- `evaluation.csv` is a CSV in which to dump the evaluation: for each dimension of each `t`, the total complexity over this dimension and the number of points not predicted within the threshold,
- `models0.txt models1.txt models2.txt` are files in which models were dumped by `cocotte learn`
and which should be evaluated.

## Using COCOTTE as a library
#### Files
- Headers for COCOTTE are in `maindirectory/include/cocotte`
- Link you code with `maindirectory/lib/libcocotte.a`
- You might find `maindirectory/include/datasources` and `maindirectory/lib/libcocotte.a` useful, as they are headers and binaries for loading data from a data file and a data structure file.

#### Data types
Two data types are defined in `maindirectory/include/cocotte/datatypes.h`:
- a `Cocotte::Measure` contains a value and a precision,
- a `Cocotte::DataPoint` contains two vectors `x` and `t`.
For each dimension `i` of `t`, COCOTTE will search for the smooth components of a function `t[i] = F_i(x)`,
where `t[i]` and `x` are vectors of measures.

#### Cocotte::Learner
`COCOTTE` is primarily used via the class `Cocotte::Learner`.
Please look at `maindirectory/src/main.cpp` for an example of how it is used.

##### Construction/dumping
- The main constructor takes the names of "inputs" (i.e. `x`) and "outputs" (i.e. `t`) as arguments.
Those names are given as vectors, with the same structure as in `Cocotte::DataPoint`.
- Models can be dumped with `dumpModels()`.
- The other constructor reconstructs it from the dumped file.
##### Adding training points
- Training points can be added in any order with `addDataPoint()` and `addDataPoints()`.
- After adding all training datapoints, call `removeArtifacts()` to eliminate the artifacts of the merging phase.
- Changes made by `removeArtifacts()` can be rolled back by `restructureModels()`.

##### Various ways to add training points
- `addDataPointToExistingModels()` and `addDataPointsToExistingModels()` uses new training datapoints to refine existing models rather than improving the model collection as a whole (which is faster).
> Please note that `restructureModels()` actually leaves `Cocotte::Learner` in the same state as if all datapoints
> were just added with `addDataPoints()`. As such, `addDataPointsToExistingModels()` can be seen as a way
> to delay computation.
- `addDataPointIncremental()` and `addDataPointsIncremental()` behave
as if they called `restructureModels()` then  `addDataPoints()`.

> As of now, the incremental version of COCOTTE is still buggy
> (it does not always return the same final models depending on the order in which the points are added),
> so we recommend adding all points at once with addDataPoints().

##### Prediction
The `predict()` function estimates the "outputs" (i.e. `t`) from the "inputs" (i.e. `x`).
It can also return the IDs of the models used for estimating those points (i.e. the result of the classifier),
which can be useful for visualizing what has been learned.

## Building on COCOTTE
COCOTTE has been implemented with some modularity in mind,
which is why `Cocotte::Learner` is templated by a type of approximator functions.
So far, the only type of approximator in the implementation is polynomial.
If you want to use another kind of approximator,
simply derive a class from `Cocotte::Approximators::Approximator`,
using `Cocotte::Approximators::Polynomial` as an example,
and use your class as a template argument for `Cocotte::Learner`.


> If you want to replace part of COCOTTE's implementation but find that COCOTTE is not modular
> enough to allow you to do that simply, don't hesitate to send me a message/raise an issue
> and I'll see if I can add some more modularity.

## Simulation data
The data I used in my paper was generated with the simulation environment that I got from people at ISIR.
While I won't put their code here, I uploaded the [data I used][data],
which consists in a [training data file][training], a [test data file][test], and [data structure file][structure].

[data]:https://github.com/AdrienMatricon/data/tree/master/2016-07-12---simulation-data-for-cocotte
[training]:https://raw.githubusercontent.com/AdrienMatricon/data/master/2016-07-12---simulation-data-for-cocotte/training.csv
[test]:https://github.com/AdrienMatricon/data/raw/master/2016-07-12---simulation-data-for-cocotte/test.csv
[structure]:https://raw.githubusercontent.com/AdrienMatricon/data/master/2016-07-12---simulation-data-for-cocotte/structure.txt


