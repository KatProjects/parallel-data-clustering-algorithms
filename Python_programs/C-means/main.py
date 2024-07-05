import numpy as np
import matplotlib.pyplot as plt

# Параметры:
num_points = 100  # количество точек
num_clusters = 3  # количество кластеров
exponential_weight = 2  # экспоненциальный вес - параметр нечёткости (по умолчанию для баланса принято 2)
epsilon = 1e-4  # пороговое значение для остановки алгоритма


# функция для отображения точек в виде графика в исходном виде одним цветом:
def start_graph(data):
    dimensions = data.shape[1]
    if dimensions == 1:
        x = data[:, 0]
        plt.hlines(0, xmin=min(x) - 1, xmax=max(x) + 1)
        plt.scatter(x, [0] * len(x), s=30)
        plt.title("Массив 100 точек в одномерном пространстве")
        plt.xlabel("Значение")
        plt.ylabel("Ось")
        plt.grid(True)
        plt.yticks([])
        plt.show()
    elif dimensions == 2:
        x = data[:, 0]
        y = data[:, 1]
        plt.scatter(x, y, s=30)
        plt.title("Массив 150 точек в двухмерном пространстве")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.show()
    elif dimensions == 3:
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        fig = plt.figure()
        plt.title("Массив 100 точек в трёхмерном пространстве")
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, s=30)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
    else:
        print("Массив не является одномерным, двумерным или трехмерным\n")


def plot_with_centers(data, centers, membership):
    dimensions = data.shape[1]
    if dimensions == 1:
        x = data[:, 0]
        plt.hlines(0, xmin=min(x) - 1, xmax=max(x) + 1)
        plt.scatter(x, [0] * len(x), c=membership, cmap='plasma', alpha=0.70, s=30)
        plt.title("Fuzzy C-means 1D")
        plt.xlabel("Значение")
        plt.ylabel("Ось")
        plt.grid(True)
        plt.scatter(centers, [0] * len(centers), c='black', marker='x', label='Centers', s=30)
        plt.legend()
        plt.xticks(np.arange(min(x), max(x) + 1, 1))
        plt.yticks([])
        plt.show()
    elif dimensions == 2:
        x = data[:, 0]
        y = data[:, 1]
        plt.scatter(x, y, c=membership, cmap='plasma', alpha=0.70, s=30)
        plt.title("Fuzzy C-means, Centers=3")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', label='Centers', s=60)
        plt.legend(loc='upper left')
        plt.show()
    elif dimensions == 3:
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        fig = plt.figure()
        plt.title("Fuzzy C-means 3D")
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c=membership, cmap='plasma', alpha=0.7, s=30)
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c="black", marker='x', label='Centers', s=30)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.grid(True)
        plt.legend()
        plt.show()
    else:
        print("Массив не является одномерным, двумерным или трехмерным\n")

# функция для выбора первых центроид случайным образом
def start_centers(data, k):
    # Нахождение минимального и максимального значения для каждого признака
    min_values = np.min(data, axis=0)
    max_values = np.max(data, axis=0)

    # Инициализация массива для хранения центров кластеров
    centers = np.zeros((k, data.shape[1]))

    # Выбор случайных центров кластеров
    for i in range(k):
        # Случайный выбор значений для каждого признака в пределах интервала
        center = np.random.uniform(min_values, max_values)  # генерация случайных значений с равномерным распределением
        centers[i] = center

    return centers


# функция для генерации начальных значений матрицы степеней принадлежности
def start_membership(data, k):
    n = data.shape[0]
    # генерация матрицы принадлежности со случайными значениями в интервале от 0 до 1 с равномерным распределением
    membership_matrix = np.random.rand(n, k)
    return membership_matrix


# пересчёт центров
def update_centers(centers, data, k, membership_matrix, m):
    n = data.shape[0]
    for j in range(k):
        w = membership_matrix ** m
        numerator = np.zeros(data.shape[1])
        denominator = 0
        for i in range(n):
            numerator += w[i, j] * data[i]
            denominator += w[i, j]
        centers[j] = numerator / denominator
    return centers


# пересчёт матрицы степеней принадлежности на основе вычисленных центроид
def update_membership(data, centers, k, m):
    n = data.shape[0]

    distances = np.zeros((n, k))  # массив для хранения расстояний
    # вычисление расстояний от каждой точки до каждого центра
    for i, point in enumerate(data):
        for j, center in enumerate(centers):
            distances[i, j] = np.sqrt((np.sum((point - center) ** 2)))

    membership_matrix = np.zeros((n, k))  # массив для хранения значений матрицы степеней принадлежности

    # вычисление степеней принадлежности для каждой точки
    for i in range(n):
        for j in range(k):
            sum_distances = 0
            for c in range(k):
                distances_ratio = distances[i, j] / distances[i, c]
                sum_distances += distances_ratio ** (2 / (m - 1))
            membership_matrix[i, j] = 1 / sum_distances

    return membership_matrix


# основная функция работы метода C-means (цикл, условие остановки, вывод центров и матрицы принадлежности)
def c_means(data, eps, m, k, max_iterations=100):
    centers = start_centers(data, k)  # инициализирую центры случайными значениями
    membership = start_membership(data, k)  # инициализирую случайную матрицу принадлежности

    # смотрю график и начальные центры
    start_graph(data)

    # цикл
    iteration = 0
    while iteration < max_iterations:
        new_membership = update_membership(data, centers, k, m)  # обновление матрицы принадлежности
        old_centers = np.copy(centers)  # сохранение копии текущих центров
        new_centers = update_centers(centers, data, k, new_membership, m)  # обновление центров
        # условие остановки алгоритма
        if np.all(np.abs(new_centers - old_centers) < eps):
            print("Алгоритм сошёлся на итерации {}".format(iteration))
            plot_with_centers(data, centers, membership)
            break
        centers = new_centers
        membership = new_membership
        # plot_with_centers(data, centers, membership)
        iteration += 1

    return centers, membership


dataPoints = np.loadtxt('my_mass100.txt', dtype=float)
result_centers, result_membership = c_means(dataPoints, epsilon, exponential_weight, num_clusters)

