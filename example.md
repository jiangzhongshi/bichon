---
title: example
layout: jekyll-theme-minimal
filename: example.md
--- 

# Hello Bunny
After compilation, you can launch (from the `build/` directory)
```bash
wget https://raw.githubusercontent.com/libigl/libigl-tutorial-data/master/bunny.off # get the bunny mesh
./cumin_bin -i bunny.off -o ./
```
To obtain `bunny.off.h5` in the current directory.

## Conversion
We provide a simple script to convert from our format to be compatible with [gmsh](https://gmsh.info/doc/texinfo/gmsh.html) visualizer for a visualization similar to Fig.4 in our paper.

Requiring python packages `pip install meshio h5py numpy`
```bash
python ../python/format_utils.py bunny.off.h5 bunny.msh
```

# Visualize
The recommended way to visualize the output file is using the python

Alternatively, you can convert it to GMSH format `.msh` through `meshio`

# Setting Constraint Points
TODO

# Setting Feature Tagging
TODO
