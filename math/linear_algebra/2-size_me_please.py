#!/usr/bin/env python3
"""
matrix_shape = __import__('2-size_me_please').matrix_shape

mat1 = [[1, 2], [3, 4]]
print(matrix_shape(mat1))
mat2 = [[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        [[16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]]
        print(matrix_shape(mat2))
        """


        def matrix_shape(matrix):
                """Calculates the shape of a matrix.

                    Args:
                            matrix: The matrix to calculate the shape of.

                                Returns:
                                        A list containing the shape of the matrix in each dimension.
                                        """

                                            shape = []
                                                while isinstance(matrix, list):
                                                            shape.append(len(matrix))
                                                                    matrix = matrix[0]  # Move one level deeper into the matrix
                                                                        return shape  # Return the shape after calculating all dimensions

