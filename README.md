# Bijective and Coarse High-Order Tetrahdedral Meshes
Zhongshi Jiang, Ziyi Zhang, Yixin Hu, Teseo Schneider, Denis Zorin, Daniele Panozzo. 
*ACM Transactions on Graphics (SIGGRAPH 2021)*

[:tv: Talk (YouTube)](https://youtu.be/yfztQw78gnE), [:memo: Paper](https://cs.nyu.edu/~zhongshi/files/bichon_preview.pdf)
## TL;DR
- :triangular_ruler: Input: Manifold and watertight triangle mesh, without self intersections.
	- Optional: Constraint Points and Feature Tagging.
- :curly_loop: Output: High Order (Bezier) Tetrahedral Mesh
	- Coarsened.
	- Valid (non-flip) Elements.
	- No Intersection.
	- Distance Bound.
	- Feature Preserving.
	- Bijective High-order Surface to the input. 
## News
- :newspaper: 06/17/2021: First version of Bichon code is released!
## Tips
- 📌 
## Dataset
:confetti_ball: We provide generated high order tetrahedral meshes and their surface for 
futher research and developement.
- [Quartic Tetrahedral Meshes (.msh)](https://drive.google.com/file/d/1Gw3vza0GkY0pMf4kLcrOzQeCIlbEp4Cs/view?usp=sharing)

## Installation via CMake [![CMake](https://github.com/jiangzhongshi/bichon/actions/workflows/cmake.yml/badge.svg)](https://github.com/jiangzhongshi/bichon/actions/workflows/cmake.yml)
Our system is developed in a Linux environment, with GCC-9 and Clang-12, and is tested on macOS and Windows (currently failing CI). 
Please refer to [cmake.yml](.github/workflows/cmake.yml) for a more detailed setup.

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ../
make -j4
```

## Usage
### Input Format
Basic input is the triangle mesh, in the format of `.obj/.off/.ply/.stl`. 
- We ask the mesh to be manifold, watertight, no degenerate triangle, and without self intersection. The precondition checks inside the program include numerical tolerance.

Optional input is and HDF5 file, encoding feature tagging and constraint points, in the format of `.h5`. 
- File might contains the following fields (all optional),
  - `E` is feature edges. Matrix of `|E|x2`, with each row indicating the endpoints (`v0,v1`) of the marked edge.
  - `V` is the additional feature corners (the junction of several features). Corners can be automatically infered if there are three or more features meet. Therefore, `V` is only supplied when there is a corner with two feature lines meet. Vector of `|V|` with vertex indices.
  - `P_fid` and `P_bc` for the constraint points where the distance bound is imposed upon. Vector of `|P|` for the list of faces where the points are, and Matrix of `|P|x3` for their corresponding barycentric coordinates on each points.
- We typically generate these files with [h5py](https://pypi.org/project/h5py/) or [HighFive (C++)](https://github.com/BlueBrain/HighFive).

### Output Format
Our output file is in HDF5 format `.h5`, with the following fields
- `lagr` as `|l|x3` matrix of *volume Lagrange* control points for the volume.
- `cells` as `|t|x|n|` as the matrix of connectivity.

Additionally, the following fields are useful in other cases.
- `complete_cp` for the *surface B\'ezeir* control points. Array of `|F| x |n| x 3` stores (in duplicates) the control points for each face.
- `mV`,`mbase`,`mtop`, `mF` encode the internal shell structure.

Our internal ordering of the node (inside each high order element) is generated recursively ([tuple_gen](python/curve/fem_generator.py:L86)).
Conversion convention to GMSH (`triangle6`, `triangle10`, `triangle15`,`tetra20`, `tetra35`) is manually coded in the same file. Further conversion is still on the way and PRs are welcome.
### Command Line Usage
```bash
./cumin_bin -i INPUT_MESH -o OUTPUT_FOLDER/
```

```
Options:
  -h,--help                   Print this help message and exit
  -i,--input TEXT:FILE REQUIRED
                              input mesh name
  -g,--graph TEXT             feature graph and constraint point file .h5
  -o,--output TEXT=./         output dir to save the serialization .h5 file
  -l,--logdir TEXT            log dir
  --curve-distance_threshold FLOAT distance bound for the sampled point. Default on all the vertices, can be specified for the feature h5.
  --curve-order INT          the order of the surface mesh. Tetrahedral mesh will be of order +1
  --feature-dihedral_threshold FLOAT automatic detecting feature based on dihedral angle.
  --shell-target_edge_length FLOAT target edge length, only as a heuritic upper bound.
```

## Visualization


`python/` folder contains several scripts to visualize or convert our serialization file.

## Examples

## License
The source code in this repository is released under MIT License. However, be aware that several dependencies (notably, CGAL with GPLv3) have differing licenses.

## Nerd Corner
- If you are interesting in our algorithm,
please refer to the papers [Bijective and Coarse High-Order Tetrahedral Meshes](https://cs.nyu.edu/~zhongshi/files/bichon_preview.pdf), and its successor [Bijective Projection in a Shell](https://dl.acm.org/doi/abs/10.1145/3414685.3417769).