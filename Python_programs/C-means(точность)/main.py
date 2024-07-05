import numpy as np

dim = 3


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
    # plot_with_centers(data, centers)

    # цикл
    iteration = 0
    while iteration < max_iterations:
        new_membership = update_membership(data, centers, k, m)  # обновление матрицы принадлежности
        old_centers = np.copy(centers)  # сохранение копии текущих центров
        new_centers = update_centers(centers, data, k, new_membership, m)  # обновление центров
        # условие останова
        if np.all(np.abs(new_centers - old_centers) < eps):
            print("Алгоритм сошёлся на итерации {}".format(iteration))
            break  # Останавливаем алгоритм
        centers = new_centers
        membership = new_membership
        # plot_with_centers(data, centers)
        iteration += 1

    real_centers = a2 = np.array([[0, 0, 0], [3, 3, 3], [-3, -3, -3]])

    # Сравнение вычисленных центров с реальными
    print("Epsilon = {}".format(eps))

    for i in range(k):
        min_dist = float("inf")
        for j in range(len(real_centers)):
            dist = np.linalg.norm(centers[i] - real_centers[j])
            min_dist = min(min_dist, dist)
        print("Центр {}, евкл: {}".format(centers[i], min_dist))

    return centers, membership


# считывание данных из файла в массив

dataPoints = np.loadtxt('my_mass100.txt', dtype=float)


#  Замеры времени
print("Замер точности работы алгоритма C-means, Python")

# c_means(dataPoints, 100, 2, 3, 100)
# c_means(dataPoints, 0.1, 2, 3, 100)
# c_means(dataPoints, 0.01, 2, 3, 100)
c_means(dataPoints, 0.001, 2, 3, 100)
c_means(dataPoints, 0.0001, 2, 3, 100)
c_means(dataPoints, 0.00001, 2, 3, 100)
c_means(dataPoints, 0.000001, 2, 3, 100)
# c_means(dataPoints, 0.0000001, 2, 3, 100)
# c_means(dataPoints, 0.00000001, 2, 3, 100)
# c_means(dataPoints, 0.000000001, 2, 3, 100)




