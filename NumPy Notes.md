# [NumPy Tutorial](https://www.w3schools.com/python/numpy/default.asp) #

## [Intro](https://www.w3schools.com/python/numpy/numpy_intro.asp) ##

- Numpy provides 50x faster array objects than Python.
- Array object is called `ndarray`
- `ndarray` is faster because it is stored in one continuous place in memory, making access and manipulation efficient; ie, `locality of reference`. NumPy is written in Python and C++ for fast computation.
- [Source Code](https://github.com/numpy/numpy)

Installation
- `pip install numpy` in terminal (with Python and pip installed)
- Import: 
  - `import numpy`
  - usually aliased: `import numpy as np`
- Hello world:
  - `import numpy
    arr = numpy.array((1,2,3])
    print(arr)`
  - `arr = np.array([4,5,6])`
  - Check version: `print(np.__version__)`

## Creating Arrays ##

Arrays are created with `array()` fn.
It accepts:
- list
- tuple
- any array-like object

These will be converted to an `ndarray`.

### O-D Array/Scalar ###
`arr = np.array(420)`
### 1-D Array ###
`arr = np.array([6,6,6])`
### 2-D Array/Matrix/Tensor ###
`arr = np.array([[4,2],[1,3]])`
### 3-D Array/3rd Order Tensor ###
An array with 2-D arrays as its elements.
` arr = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])`
### Number of Dimensions ###
`print(arr.ndim)`
### Higher dimensions ###
Five dimensions:

`arr = np.array([1,2,3,4], ndmin=5)`

``print(`num dimensions:`, arr.ndim)``


## [NumPy Array Indexing](https://www.w3schools.com/python/numpy/numpy_array_indexing.asp) ##
### Access Array Elements ###
`print(arr[0])`
### 2-D access ###
`print('5th element on 2nd dim: ', arr[1, 4])`
### 3-D ###
`print(arr[0, 1, 2])`
### Negative indexing ###
`print('Last element from 2nd dim: ', arr[1, -1])`
## [NumPy Array Slicing](https://www.w3schools.com/python/numpy/numpy_array_slicing.asp) ##
`arr[start:end]`
`arr[start:end:step]`

Slice from index n to the end:
`arr[n:]`

Slice from beginning to index n (not included):
`arr[:4]`
