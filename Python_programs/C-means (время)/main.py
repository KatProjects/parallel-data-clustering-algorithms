import numpy as np
import time

# количество итераций для замера времени
numIterations = 100


# функция для выбора первых центроид случайным образом
def start_centers(data, k):
    # Нахождение минимального и максимального значения для каждого признака
    min_values = np.min(data, axis=0)
    max_values = np.max(data, axis=0)

    # Инициализация массива для хранения центров кластеров
    centers = np.zeros((k, data.shape[1]))

    # Выбор случайных центров кластеров
    for i in range(k):
        # Случайный выбор значений для каждого признака в пределах соответствующего интервала
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
def c_means(data, eps, m, k, max_iterations=10):
    # n = data.shape[0]
    centers = start_centers(data, k)  # инициализирую центры случайными значениями
    membership = start_membership(data, k)  # инициализирую случайную матрицу принадлежности

    # цикл
    iteration = 0
    while iteration < max_iterations:
        new_membership = update_membership(data, centers, k, m)  # обновление матрицы принадлежности
        new_centers = update_centers(centers, data, k, new_membership, m)  # обновление центров
        # условие останова
        # if np.all(np.abs(new_centers - old_centers) < eps):
        #     print("Алгоритм сошёлся на итерации {}".format(iteration))
        #     break
        centers = new_centers
        membership = new_membership
        iteration += 1

    return centers, membership


# считывание данных из файла в массив
dataPoints = np.loadtxt('my_mass100000.txt', dtype=float)


#  Замеры времени
print("Замер времени работы алгоритма C-means на 100 итерациях.")
print(time.ctime())

total_execution_time = 0
for i in range(numIterations):
    start_time = time.time()  # начальная точка замера времени

    # измеряемый код
    c_means(dataPoints, 1e-4, 2, 3, 10)

    end_time = time.time()  # конечная точка замера времени
    execution_time = end_time - start_time  # вычисление времени работы программы
    total_execution_time += execution_time

average_execution_time = total_execution_time / numIterations

# вывод времени работы программы в секундах
print("Результат для массива (100000, 3), Python, CPU: {} секунд".format(average_execution_time))