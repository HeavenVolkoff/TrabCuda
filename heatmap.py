#!python

import matplotlib.pyplot as plt

with open("matrix.csv", "r") as f:
	text = f.read()

matrix = text.split("\n")
for i in range(len(matrix)):
	if(matrix[i] == ' ' or matrix[i] == '\n'):
		continue
	matrix[i] = matrix[i].split(", ")
	for j in range(len(matrix[i])):
		matrix[i][j] = float(matrix[i][j])

plt.imshow(matrix, cmap='hot', interpolation='nearest', origin='lower')
plt.show()