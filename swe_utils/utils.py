import matplotlib.pyplot as plt
import numpy as np
def get_trajectories_from_list(value_list, dim1, dim2, dim3):
    lists = [[],[],[]]
    for element in value_list:
        lists[0].append(element.flatten()[dim1])
        lists[1].append(element.flatten()[dim2])
        lists[2].append(element.flatten()[dim3])
    for i in range(3):
        lists[i] = np.stack(lists[i])
    return lists
def plot_trajectories(name, list1, list2, dim1, dim2, dim3):
    points = get_trajectories_from_list(list1, dim1, dim2, dim3)
    points_2 = get_trajectories_from_list(list2, dim1, dim2, dim3)
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111, projection = '3d')
    ax.plot(points[0], points[1], points[2], marker = 'x', color = 'red')
    ax.scatter(points[0], points[1], points[2], color = 'red')
    ax.plot(points_2[0], points_2[1], points_2[2], marker = 'x', color = 'blue')
    ax.scatter(points_2[0], points_2[1], points_2[2], color = 'blue')
    plt.savefig(name)

def show_trajectories(list1, list2, dim1, dim2, dim3):
    points = get_trajectories_from_list(list1, dim1, dim2, dim3)
    points_2 = get_trajectories_from_list(list2, dim1, dim2, dim3)
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111, projection = '3d')
    ax.plot(points[0], points[1], points[2], marker = 'x', color = 'red')
    ax.scatter(points[0], points[1], points[2], color = 'red')
    ax.plot(points_2[0], points_2[1], points_2[2], marker = 'x', color = 'blue')
    ax.plot(points_2[0][0], points_2[1][0], points_2[2][0], marker = 'o', markersize=22, color = 'blue')
    ax.plot(points[0][0], points[1][0], points[2][0], marker = 'o', markersize=22, color = 'red')
    ax.scatter(points_2[0], points_2[1], points_2[2], color = 'blue')
    plt.show()