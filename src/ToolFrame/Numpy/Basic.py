import numpy as np

# a = np.array([1,2,3])
# print(type(a))
# print(a.shape)
# b = np.array([[1,2,3],[4,5,6]])
# print(b.shape)
# a = np.zeros((2,2))
# print("np.zeros(2,2)")
# print(a)
# b = np.ones((1,2))
# print("np.ones((1,2))")
# print(b)
# c = np.full((2,2),7)
# print("np.full((2,2),7)")
# print(c)
# d = np.eye(2)
# print("np.eye(2)")
# print(d)
# e = np.random.random((2,2))
# print("np.random.random((2,2))")
# print(e)

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
# a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
#
# # Use slicing to pull out the subarray consisting of the first 2 rows
# # and columns 1 and 2; b is the following array of shape (2, 2):
# # [[2 3]
# #  [6 7]]
# b = a[:2, 1:3]
#
# # A slice of an array is a view into the same data, so modifying it
# # will modify the original array.
# print(a[0, 1])   # Prints "2"
# b[0, 0] = 77     # b[0, 0] is the same piece of data as a[0, 1]
# print(a[0, 1])   # Prints "77"
#
# a = np.array([[1,2], [3, 4], [5, 6]])
#
# # An example of integer array indexing.
# # The returned array will have shape (3,) and
# print(a[[0, 1, 2], [0, 1, 0]])  # Prints "[1 4 5]"
#
# # The above example of integer array indexing is equivalent to this:
# print(np.array([a[0, 0], a[1, 1], a[2, 0]]))  # Prints "[1 4 5]"
#
# # When using integer array indexing, you can reuse the same
# # element from the source array:
# print(a[[0, 0], [1, 1]])  # Prints "[2 2]"
#
# # Equivalent to the previous integer array indexing example
# print(np.array([a[0, 1], a[0, 1]]))  # Prints "[2 2]"

a = np.array([[1,2], [3, 4], [5, 6]])

# An example of integer array indexing.
# The returned array will have shape (3,) and
print(a[[0, 1, 2], [0, 1, 0]])  # Prints "[1 4 5]"

# The above example of integer array indexing is equivalent to this:
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))  # Prints "[1 4 5]"

# When using integer array indexing, you can reuse the same
# element from the source array:
print(a[[0, 0], [1, 1]])  # Prints "[2 2]"

# Equivalent to the previous integer array indexing example
print(np.array([a[0, 1], a[0, 1]]))  # Prints "[2 2]"
