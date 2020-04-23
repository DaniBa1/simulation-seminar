import matplotlib.pyplot as plt
import numpy as np


def ring_matrix(n):
    matrix = np.zeros((n, n))
    for state in range(n):
        # left
        matrix[state][(state - 1) % n] = 0.5
        # right
        matrix[state][(state + 1) % n] = 0.5
    return matrix


def get_errorplot(n, transition_matrix, stationary_distribution, start_distribution=None):
    if start_distribution is None:
        size = transition_matrix.shape[0]
        start_distribution = np.random.random((1, size))[0]
        start_distribution /= sum(start_distribution)

    error_vector = []
    current_distribution = start_distribution
    error_vector.append(np.linalg.norm(current_distribution - stationary_distribution))

    for i in range(n):
        current_distribution = current_distribution.dot(transition_matrix)
        # print(current_distribution)
        error_vector.append(np.linalg.norm(current_distribution - stationary_distribution))

    plt.plot(error_vector)
    plt.ylim([0, max(error_vector)])
    print(max(error_vector))
    plt.xlim([0,n])
    plt.show()




reps = 1000
size = 15
t_matrix = ring_matrix(size)
statitonary_dist = np.ones(size) / size
start_dist = np.eye(size)[0]
get_errorplot(reps, t_matrix, statitonary_dist,start_dist)
