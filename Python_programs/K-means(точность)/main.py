import numpy as np

def starting_centers(data_points, k):
    # Нахождение минимального и максимального значения для каждого признака
    min_values = np.min(data_points, axis=0)
    max_values = np.max(data_points, axis=0)

    # Инициализация массива для хранения центров кластеров
    centers = np.zeros((k, data_points.shape[1]))

    # Выбор случайных центров кластеров
    for i in range(k):
        # Случайный выбор значений для каждого признака в пределах интервала
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
    print("\nEpsilon = {}".format(eps))
    # генерация случайных 3-х начальных центров
    centers = starting_centers(points_data, k)

    # Итерации алгоритма
    iteration = 0  # инициализация стартового значения
    clusters = [[]]
    # цикл
    while iteration < max_iterations:
        # приближение данных к центрам, формирование кластеров
        clusters = cluster_distribution(points_data, centers)

        # пересчёт новых центров кластеров
        new_centers = recalculation_centers(clusters)

        # условие останова, если модуль всех расстояний меньше эпсилон - завершаем работу алгоритма
        try:
            if np.all(np.abs(new_centers - centers) < eps):
                print("Алгоритм сошёлся на итерации {}".format(iteration))
                break
        except ValueError as err:
            print("Размерность кластеров уменьшилась на итерации {}".format(iteration))

        centers = new_centers
        iteration += 1

    real_centers = a2 = np.array([[0, 0, 0], [3, 3, 3], [-3, -3, -3]])

    # Сравнение вычисленных центров с реальными
    try:
        for i in range(k):
            min_dist = float("inf")
            for j in range(len(real_centers)):
                dist = np.linalg.norm(centers[i] - real_centers[j])
                min_dist = min(min_dist, dist)
            print("Центр {}, евкл: {}".format(centers[i], min_dist))
    except IndexError as err:
        return centers

    return centers, clusters


dataPoints = np.loadtxt('my_mass100.txt', dtype=float)


#  Замеры времени
print("Замер точности работы алгоритма C-means, Python")

k_means(dataPoints, 3, 100, 100)
k_means(dataPoints, 3, 100, 0.1)
k_means(dataPoints, 3, 100, 0.001)
k_means(dataPoints, 3, 100, 0.0001)
k_means(dataPoints, 3, 100, 0.00001)
k_means(dataPoints, 3, 100, 0.000001)
k_means(dataPoints, 3, 100, 0.0000001)

