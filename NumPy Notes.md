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

Negative Slicing (index 3 to index 1 from the **end**): `arr[-3:-1]`

Stepping. 
- Returns every other element from index 1 to 5: `arr[1:5:2]`
- Returns every other element from entire array: `arr[::2]`
2-D Arrays:
- 2nd element, from index 1 to 4 `arr[1,1:4]`
- Elements from first two indexes `arr[0:2, 1:4]`

## [Data Types](https://www.w3schools.com/python/numpy/numpy_data_types.asp) ##
Python datatypes:
  - string
  - integer
  - float
  - boolean
  - complex (e.g.: 1.0+2.0j)

NumPy Data Types:
  - i - integer
  - b - boolean
  - u - unsigned integer
  - f - float
  - c - complex float
  - m - timedelta
  - M - datetime
  - O - object
  - S - string
  - U - unicode string
  - V - fixed chunk of memory for other type ( void )

Data type operations:
  - Return data type of an array object: `print(arr.dtype)`
  - Creating array with defined data type (string): 
    - `arr = np.array([1,2,3,4], dtype='S')`
  - Create array with data type 4-bytes-integer:
    - `arr = np.array([1, 2, 3, 4], dtype='i4')`
  - To change data type of an array, make a copy using `astype()`. (Ex: float to integer)
    ```python
      arr = np.array([1.1, 2.1, 3.1])
      newarr = arr.astype('i')
    ```

## [Copy vs View](https://www.w3schools.com/python/numpy/numpy_copy_vs_view.asp) ##
  - Copy makes a new array from another
  - View is similar to a reference to the original array.
    ```python
      arr = np.array([1, 2, 3, 4, 5])
      x = arr.copy()
      y = arr.view()
      x[0] = 6 # does not modify arr
      y[0] = 9 # modifies arr
    ```
  - Copy owns data, view does not own data
  - Check using `base`. Returns None if array owns the data
    ```python
      print(x.base) # copy, returns None
      print(y.base) # view, returns original array
    ```

## [Array Shape](https://www.w3schools.com/python/numpy/numpy_array_shape.asp) ##
- NumPy arrays have attribute called `shape`, returns tuple with "dimensions" and number of elements.
  - ```python 
    arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    print(arr.shape) # returns: (2,4)
    ```

## [Array Reshape](https://www.w3schools.com/python/numpy/numpy_array_reshape.asp) ##
- add or remove dimensions; or, change num of elements in ea dimension
  ```python
  arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
  # converts 1-D array of 12 element into 2-D array
  # outermost dim with 4 arrays, ea with 3 elements:
  newarr = arr.reshape(4, 3) 
  ```
  ```python
  arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
  # converts 1-D array of 12 element into 3-D array
  # outermost dim with 2 arrays, ea with 3 arrays of 2 elements ea:
  newarr = arr.reshape(2, 3, 2)
  ```
- num of elements has to be the same in the reshape
- `print(arr.reshape(2, 4).base)` returns og array (it's a view)
  
Unknown Dimension
- one unknown dimension allowed, which doesn't have specified dimensions
- NumPy will calculate the dims
- Ex converts 1D array w/ 8 eles to 3D array with 2x2 eles
  ```python
  arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
  newarr = arr.reshape(2, 2, -1)
  ```
Flatten array:
- Converts array to 1D array:
  ```python
  arr = np.array([[1, 2, 3], [4, 5, 6]])
  newarr = arr.reshape(-1)
  ```

## [Iterating Arrays](https://www.w3schools.com/python/numpy/numpy_array_iterating.asp) ##

- Using a for loop over the array will print the contained elements or arrays. 
  ```python
  arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

  for x in arr:
    print(x)
  ```
- To print individual elements in 2D+ arrays, iterate in each dimension.
  ```python
  for x in arr:
    for y in x:
      for z in y:
       print(z)
  ```
nditer()
  - fn `nditer()` for basic to advanced iterations
  - ex: prints each element
  ```python
  arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

  for x in np.nditer(arr):
    print(x)
  ```

Change datatypes while iterating:
- use `op_dtypes` to change datatypes of elements while iterating
- NumPy won't change data type in-place, so use buffer
  ```python
  arr = np.array([1, 2, 3])

  for x in np.nditer(arr, flags=['buffered'], op_dtypes=['S']):
    print(x)
  ```
- can also use step size with nditer: 
  ```python
  for x in np.nditer(arr[:,::2]):
    print(x)
  ```
To enumerate/count while iterating, use `ndenumerate()`:
  ```python
    arr = np.array([1, 2, 3])

  for idx, x in np.ndenumerate(arr):
   print(idx, x) 
   ```

## [Joining Array](https://www.w3schools.com/python/numpy/numpy_array_join.asp) ##

- In NumPy, arrays are joined by axes.
- Use `concatenate()` on arrays to join along with axis.
- Without explicit axis, defaults to 0.
- 1D arrays
  ```python
  arr1 = np.array([1, 2, 3])
  arr2 = np.array([4, 5, 6])
  arr = np.concatenate((arr1, arr2))
  ```
  `[1 2 3 4 5 6]`
- 2D arrays:
    ```python
    arr1 = np.array([[1, 2], [3, 4]])
    arr2 = np.array([[5, 6], [7, 8]])
    arr = np.concatenate((arr1, arr2), axis=1)
    ```
    `[[1 2 5 6]
      [3 4 7 8]]`

Stacking
  - Same as concatenation, but performed along a new axis.
    ```python
      arr1 = np.array([1, 2, 3])
      arr2 = np.array([4, 5, 6])
      arr = np.stack((arr1, arr2), axis=1)
      print(arr)
    ```
    ```
    [[1 4]
     [2 5]
     [3 6]]
    ```
  - Stacking Along Rows using `hstack()`, 
    ```python
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])
    arr = np.hstack((arr1, arr2))
    print(arr)
    ```
    `[1 2 3 4 5 6]`
  - Stack along columns using `vstack()`
    
    `arr = np.vstack((arr1, arr2))`
    ``` 
    [[1 2 3]
    [4 5 6]] 
    ```
  - Stack along height/depth 
  
    `arr = np.dstack((arr1, arr2))` 
    ```
    [[[1 4]
    [2 5]
    [3 6]]]
    ```

## [Array Split](https://www.w3schools.com/python/numpy/numpy_array_split.asp) ##
  - Pass `array_split()` the array and num of splits
    
    `newarr = np.array_split(arr, 3)`
  Returns an array with the split arrays as elements
  - Works with axis or hsplit
    ```python
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13,     14, 15], [16, 17, 18]])
    newarr = np.array_split(arr, 3, axis=1)
    # or:
    # newarr = np.hsplit(arr, 3)
    ```
    ```
    [array([[ 1],
          [ 4],
          [ 7],
          [10],
          [13],
          [16]]), array([[ 2],
          [ 5],
          [ 8],
          [11],
          [14],
          [17]]), array([[ 3],
          [ 6],
          [ 9],
          [12],
          [15],
          [18]])]
      ```
## [Array Search](https://www.w3schools.com/python/numpy/numpy_array_search.asp) ##
- `where(arr==4)` will return a tuple with indices where element==4.
- When adding a new element to an array, use `searchsorted(arr,7)` on a sorted array to find the appropriate sorted index to insert 7.
- Also works from right side: `x = np.searchsorted(arr, 7, side='right')`
- Works for multiple values `x = np.searchsorted(arr, [2, 4, 6])`

## [Sort](https://www.w3schools.com/python/numpy/numpy_array_sort.asp) ##
- `np.sort(arr)` returns a sorted *copy* of arr.
- Works for any data type (strings, booleans, 2-D arrays (both sort))

## [Filtering](https://www.w3schools.com/python/numpy/numpy_array_filter.asp) ## 
- NumPy filters elements out of an array to a new array using a *boolean index list*. True means element is in the filtered array, else false.
  ```python
    arr = np.array([41, 42, 43, 44])

    # Create an empty list
    filter_arr = []

    # go through each element in arr
    for element in arr:
      # if the element is higher than 42
      # set the value to True, otherwise False:
      if element > 42:
        filter_arr.append(True)
      else:
        filter_arr.append(False)

    newarr = arr[filter_arr]

    print(filter_arr)
    print(newarr)
    ```
    *OR*
    ```
    arr = np.array([41, 42, 43, 44])

    filter_arr = arr > 42

    newarr = arr[filter_arr]

    print(filter_arr)
    print(newarr)
    ```
    Returns:
    ```
    [False, False, True, True]
    [43 44]
    ```

# [Pandas](https://www.w3schools.com/python/pandas/default.asp) #
- `pip install pandas` 
- `import pandas as pd`
- `pd.__version__` returns pandas version

## [Pandas Series](https://www.w3schools.com/python/pandas/pandas_series.asp) ##
- Pandas Series is like a col in a table.
- 1D array with data of any type.
  ```python
  import pandas as pd

  a = [1, 7, 2]

  myvar = pd.Series(a)

  print(myvar)
  ```
  ```
  0    1
  1    7
  2    2
  dtype: int64
  ```
- Labels can be created with the index argument:
  ```python
  a = [1, 7, 2]

  myvar = pd.Series(a, index = ["x", "y", "z"])

  print(myvar)
  ```
  ```
  x    1
  y    7
  z    2
  dtype: int64
  ```
- Items are accessible by labels `print(myvar["y"])`

### Key/Value Objects as Series ###
- Works like a dictionary
  ```python
  calories = {"day1": 420, "day2": 380, "day3": 390}
  myvar = pd.Series(calories)
  print(myvar)
  ```
  ```
  day1    420
  day2    380
  day3    390
  dtype: int64
  ```
- Keys become labels.
- Can select subset of items using index arg: `myvar = pd.Series(calories, index = ["day1", "day2"])`

## [DataFrames](https://www.w3schools.com/python/pandas/pandas_dataframes.asp)
- Multi-dimensional/2D tables
- Series is like a col, DataFrame is like a whole table
  ```python
  data = {
    "calories": [420, 380, 390],
    "duration": [50, 40, 45]
  }
  myvar = pd.DataFrame(data)
  print(myvar)
  ```
  ```
    calories  duration
  0       420        50
  1       380        40
  2       390        45
  ```
- `loc` returns a pd Series for one row: `print(df.loc[0])`
  ```
  calories    420
  duration     50
  Name: 0, dtype: int64
  ```
- Works with more than one row, returning a DataFrame `print(df.loc[[0, 1]])`
  ```
     calories  duration
  0       420        50
  1       380        40
  ```

### Named Index
- `index` lets you name indexes
  ```python
  data = {
    "calories": [420, 380, 390],
    "duration": [50, 40, 45]
  }
  df = pd.DataFrame(data, index = ["day1", "day2", "day3"])
  print(df) 
  ```
  ```
          calories  duration
  day1       420        50
  day2       380        40
  day3       390        45
  ```
  - Can be located using `loc`: `print(df.loc["day2"])`
  ```
  calories    380
  duration     40
  Name: 0, dtype: int64
  ```

## [Loading CSV into DataFrame](https://www.w3schools.com/python/pandas/pandas_csv.asp)
- `df = pd.read_csv('data.csv')`
- `to_string()` prints entire DataFrame
- `print(df)` only returns first 5 and last 5 rows.

## [Pandas read JSON](https://www.w3schools.com/python/pandas/pandas_json.asp)
- `df = pd.read_json('data.json')`
- JSON objects have same format as Python dictionaries.
- Can be loaded into a df directly
  ```python
  data = {
    "Duration":{
      "0":60,
      "1":60,
      "2":60,
      "3":45,
      "4":45,
      "5":60
    },
    "Pulse":{
      "0":110,
      "1":117,
      "2":103,
      "3":109,
      "4":117,
      "5":102
    },
    "Maxpulse":{
      "0":130,
      "1":145,
      "2":135,
      "3":175,
      "4":148,
      "5":127
    },
    "Calories":{
      "0":409,
      "1":479,
      "2":340,
      "3":282,
      "4":406,
      "5":300
    }
  }

  df = pd.DataFrame(data)
  print(df) 
  ```
  ```
    Duration  Pulse  Maxpulse  Calories
  0        60    110       130     409.1
  1        60    117       145     479.0
  2        60    103       135     340.0
  3        45    109       175     282.4
  4        45    117       148     406.0
  5        60    102       127     300.5
  ```

## [Viewing Data](https://www.w3schools.com/python/pandas/pandas_analyzing.asp)
- `head()` method returns specified num of headers and rows (or 5 if unspecified) from the top: `print(df.head(10))`
- `tail()` works same way, but from the bottom.

### Info about data
`print(df.head(10))`
```
  <class 'pandas.core.frame.DataFrame'>
  RangeIndex: 169 entries, 0 to 168
  Data columns (total 4 columns):
   #   Column    Non-Null Count  Dtype  
  ---  ------    --------------  -----  
   0   Duration  169 non-null    int64  
   1   Pulse     169 non-null    int64  
   2   Maxpulse  169 non-null    int64  
   3   Calories  164 non-null    float64
  dtypes: float64(1), int64(3)
  memory usage: 5.4 KB
  None
```

## [Cleaning data](https://www.w3schools.com/python/pandas/pandas_cleaning.asp)
- remove bad/unwanted data, including:
  - Empty cells
  - Wrong format
  - Wrong data
  - Duplicates

### [Remove empty cells](https://www.w3schools.com/python/pandas/pandas_cleaning_empty_cells.asp)
- Remove empty rows with `dropna()`, returning a *new* df.
  ```python
  df = pd.read_csv('data.csv')
  new_df = df.dropna()
  print(new_df.to_string())
  ```
- Remove NULL vals from original df with: `df.dropna(inplace = True)` 
- Replace empty vals with a new val: e.g., replace with number 130
  `df.fillna(130, inplace = True)`
- To target specific columns: e.g., Calories col `df["Calories"].fillna(130, inplace = True)`
- pd uses `mean()`, `median()`, and `mode()`. Can be used to replace vals:
  ```python
  x = df["Calories"].mean()

  df["Calories"].fillna(x, inplace = True)
  ```

### [Cleaning Wrong Format](https://www.w3schools.com/python/pandas/pandas_cleaning_wrong_format.asp)
- Convert all cells to dates: `df['Date'] = pd.to_datetime(df['Date'])`
- Remove NULL values under dates: `df.dropna(subset=['Date'], inplace = True)`

### Wrong Data
- Replace a specific value, like a typo: `df.loc[7, 'Duration'] = 45`
- Works iteratively:
  ```python
  for x in df.index:
    if df.loc[x, "Duration"] > 120:
      df.loc[x, "Duration"] = 120
  ```
- To remove:
  ```python
  for x in df.index:
    if df.loc[x, "Duration"] > 120:
      df.drop(x, inplace = True)
  ```

### [Removing duplicates](https://www.w3schools.com/python/pandas/pandas_cleaning_duplicates.asp)
- `df.duplicated()` returns bool val for each row (True for duplicates, False otherwise).
- `df.drop_duplicates(inplace = True)` removes all duplicates from original df.
  
## [Correlations](https://www.w3schools.com/python/pandas/pandas_correlations.asp)
- `df.corr()` shows relationship between cols.
- 1 and -1 mean perfect correlation
- .6 or -.6 are good rules of thumb for decent correlation

## [Plotting data with pd](https://www.w3schools.com/python/pandas/pandas_plotting.asp)
- use `plot()` t
- Can use Pyplot from Matplotlib to visualize diagram on screen
  ```python
  import pandas as pd
  import matplotlib.pyplot as plt

  df = pd.read_csv('data.csv')

  df.plot()

  plt.show()
  ```

### Plot types
- plot type can be specified with `kind` arg
- Scatter: `df.plot(kind = 'scatter', x = 'Duration', y = 'Calories')`
- Histogram: `df["Duration"].plot(kind = 'hist')`
