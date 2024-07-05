import numpy as np
# import matplotlib.pyplot as plt
import time  # импорт библиотеки для замера времени

# параметры

# количество точек
n = 100

# размерность пространства
dim = 3

# количество итераций для замера времени
numIterations = 100

# порог для остановки
epsilon = 1e-4


# функция для выбора первых центроид случайным образом
def starting_centers(data_points, k):
    # Нахождение минимального и максимального значения для каждого признака
    min_values = np.min(data_points, axis=0)
    max_values = np.max(data_points, axis=0)

    # Инициализация массива для хранения центров кластеров
    centers = np.zeros((k, data_points.shape[1]))

    # Выбор случайных центров кластеров
    for i in range(k):
        # Случайный выбор значений для каждого признака в пределах соответствующего интервала
        center = np.random.uniform(min_values, max_values)
        centers[i] = center

    return centers


# распределение данных по кластерам, назначение точек центроидам
def cluster_distribution(data_points, centers):
    clusters = [[] for i in range(len(centers))]  # пустой список для хранения кластеров

    for x in data_points:
        distances = np.zeros(len(centers))
        for i, center in enumerate(centers):
            # евклидовы расстояния от точек до центров (индекс - метка кластера)
            distances[i] = np.sqrt(np.sum((x - center) ** 2))
        # индекс кластера с наименьшим расстоянием до точки
        cluster_index = np.argmin(distances)
        clusters[cluster_index].append(x)  # к найденному индексу добавляется текущая точка
    # возвращается список с кластерами
    return clusters


# пересчёт центров кластеров
def recalculation_centers(clusters):
    new_centers = []  # пустой массив для хранения пересчитанных центров
    for cluster in clusters:
        if len(cluster) > 0:  # проверка, не пустой ли кластер
            new_center = np.mean(cluster, axis=0)
            new_centers.append(new_center)
    return np.array(new_centers)


# Работа алгоритма
def k_means(points_data, k, max_iterations=10, eps=1e-4):

    # генерация случайных начальных центров
    centers = starting_centers(points_data, k)

    # Итерации алгоритма
    clusters = [[]]
    iteration = 0  # инициализация стартового значения
    # цикл
    while iteration < max_iterations:
        # приближение данных к центрам, формирование кластеров
        clusters = cluster_distribution(points_data, centers)

        # пересчёт новых центров кластеров
        new_centers = recalculation_centers(clusters)

        # условие останова, если модуль всех расстояний меньше эпсилон - завершаем работу алгоритма
        #try:
        #    if np.all(np.abs(new_centers - centers) < eps):
        #        print("Алгоритм сошёлся на итерации {}".format(iteration))
        #        break
        #except ValueError as err:
        #    print("Размерность кластеров уменьшилась")

        centers = new_centers
        iteration += 1

    return clusters


# считывание данных из файла в массив

data = np.loadtxt('my_mass100.txt', dtype=float)

# для массива на миллион точек
# for i in range(10):
#     filename = f"my_mass1000000_part_{i}.txt"
#     data_part = np.loadtxt(filename, dtype=float)
#     data = np.vstack((data, data_part))


#  Замеры времени
print("Замер времени работы алгоритма K-means на 10-ти итерациях.")
print(time.ctime())

total_execution_time = 0
for i in range(numIterations):
    start_time = time.time()  # начальная точка замера времени

    # измеряемый код
    k_means(data, 3, max_iterations=10)

    end_time = time.time()  # конечная точка замера времени
    execution_time = end_time - start_time  # вычисление времени работы программы
    total_execution_time += execution_time

average_execution_time = total_execution_time / numIterations

# вывод времени работы программы в секундах
print("Результат для массива (100, 3), Python, CPU: {} секунд".format(average_execution_time))