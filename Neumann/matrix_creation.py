import numpy as np

a_hat = np.array([[1, -1], [-1, 1]])
b_hat = np.array([[1 / 3, 1 / 6], [1 / 6, 1 / 3]])

rows, cols = a_hat.shape
print(f"Alpha matrix: {rows} rows, {cols} columns")
rows, cols = b_hat.shape
print(f"Beta matrix: {rows} rows, {cols} columns")
