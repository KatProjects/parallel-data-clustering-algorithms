import numpy as np
import matplotlib.pyplot as plt

# Параметры
epsilon = 1.43  # Радиус окрестности
minPts = 3   # Минимальное количество точек в окрестности


def distance(point1, point2):
    """Вычисление евклидова расстояния"""
    return np.sqrt(np.sum((point1 - point2) ** 2))


def find_neighbors(data, point_index):
    """Нахождение соседей для заданной точки"""
    neighbors = []  # пустой список для соседей

    for i in range(len(data)):
        # если точка не она сама и расстояние до исследуемой меньше эпсилон
        if i != point_index and distance(data[i], data[point_index]) <= epsilon:
            # её индекс добавляется в список соседей
            neighbors.append(i)
    return neighbors


def expand_cluster(data, start_point, cluster_id, visited, clusters):
    """Расширение кластера"""
    stack = [start_point]  # Стек для отслеживания точек, которые нужно обработать
    visited.add(start_point)  # Начальная точка добавляется в множество посещённых

    # Пока стек не пуст
    while stack:
        point_index = stack.pop()  # Точка извлекается из стека
        clusters[cluster_id].append(point_index)  # Добавляется в текущий кластер
        neighbors = find_neighbors(data, point_index)  # Осуществляется поиск соседей
        for neighbor in neighbors:
            if neighbor not in visited:
                stack.append(neighbor)  # Точка-сосед добавляется в стек
                visited.add(neighbor)  # Помечаем соседа как посещенного


def dbscan(data):
    """Реализация алгоритма DBSCAN"""
    clusters = {}  # Словарь для хранения кластеров
    cluster_id = 0  # Идентификатор текущего кластера
    visited = set()  # Множество посещенных точек
    noise = []  # Список для хранения шумовых точек

    for i, point in enumerate(data):
        if i not in visited:
            neighbors = find_neighbors(data, i)
            if len(neighbors) >= minPts:
                clusters[cluster_id] = []
                expand_cluster(data, i, cluster_id, visited, clusters)
                cluster_id += 1
            else:
                noise.append(i)  # Если точка не является точкой ядра и не попадает в кластер, она помечается как шумовая
                visited.add(i)  # Помечаем эту точку как посещенную, чтобы не обрабатывать её снова
    return clusters, noise


def stars_graph(data):
    """Визуализация исходных данных"""
    # Определение размерности данных
    dimensions = data.shape[1]

    if dimensions == 1:
        plt.figure(figsize=(10, 6))
        x = data[:, 0]
        plt.hlines(0, xmin=min(x) - 1, xmax=max(x) + 1)  # Отрисовка оси X
        plt.scatter(x, [0] * len(x), s=30, c='black', marker='.')
        plt.title("Массив 100 точек в одномерном пространстве")
        plt.xlabel("Значение")
        plt.ylabel("Ось")
        plt.grid(True)
        plt.yticks([])
        plt.show()

    elif dimensions == 2:
        plt.figure(figsize=(10, 6))
        plt.scatter(data[:, 0], data[:, 1], c='black', marker='.')
        plt.title("Массив 150 точек в двумерном пространстве")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.show()

    elif dimensions == 3:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        plt.title("Массив 100 точек в трёхмерном пространстве")
        ax.scatter(x, y, z, c='black', marker='.')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title("Массив 100 точек в трёхмерном пространстве")
        plt.show()

    else:
        print("Массив не является одномерным, двумерным или трехмерным\n")


def visualize_clusters(data, clusters, noise):
    """Визуализация результата кластеризации"""
    dimensions = data.shape[1]

    if dimensions == 1:
        plt.figure(figsize=(10, 6))
        x = data[:, 0]
        plt.hlines(0, xmin=min(x) - 1, xmax=max(x) + 1)  # Отрисовка оси X
        plt.scatter(x, [0] * len(x), s=30, c='black', marker='.', label='Шумовые точки')
        for cluster_id, cluster_points in clusters.items():
            cluster_points_data = data[cluster_points]
            plt.scatter(cluster_points_data[:, 0], [0] * len(cluster_points_data), label=f'Кластер {cluster_id}',
                        alpha=0.7, s=30)
        plt.title('DBSCAN, epsilon=0.8, minPts=5')
        plt.xlabel('Значение')
        plt.yticks([])
        plt.legend()
        plt.grid(True)
        plt.show()

    elif dimensions == 2:
        plt.figure(figsize=(10, 6))
        plt.scatter(data[:, 0], data[:, 1], c='black', marker='.')
        for cluster_id, cluster_points in clusters.items():
            cluster_points_data = data[cluster_points]
            plt.scatter(cluster_points_data[:, 0], cluster_points_data[:, 1], label=f'Cluster {cluster_id}', alpha=0.7,
                        s=30)
        plt.title('DBSCAN, epsilon=1, minPts=3')
        plt.xlabel('X')
        plt.ylabel('Y')
        #plt.legend(loc='upper left')
        plt.grid(True)
        plt.show()

    elif dimensions == 3:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        ax.scatter(x, y, z, c='black', marker='.')
        for cluster_id, cluster_points in clusters.items():
            cluster_points_data = data[cluster_points]
            ax.scatter(cluster_points_data[:, 0], cluster_points_data[:, 1], cluster_points_data[:, 2],
                       label=f'Кластер {cluster_id}', alpha=0.7, s=30)
        plt.title('DBSCAN, epsilon=1, minPts=3')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

    else:
        print("Массив не является одномерным, двумерным или трехмерным\n")


# Считывание данных
dataPoints = np.loadtxt('my_mass100.txt', dtype=float)

# Визуализация исходных данных
stars_graph(dataPoints)

# Применение алгоритма DBSCAN
clusters, noise = dbscan(dataPoints)
print("Количество найденных кластеров: {}".format(len(clusters)))

# Вывод кластеров
print("Индексы точек по кластерам: ")
for cluster_id in range(0, len(clusters)):
    print(cluster_id, ": ", sorted(clusters[cluster_id]))

print()
print("Количество шумовых точек: {}".format(len(noise)))
print("Индексы шумовых точек: {}".format(noise))


# Визуализация результатов кластеризации
visualize_clusters(dataPoints, clusters, noise)
