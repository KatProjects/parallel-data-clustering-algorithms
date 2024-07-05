import numpy as np
import matplotlib.pyplot as plt
import time

# Параметры алгоритма DBSCAN
epsilon = 1.43  # Радиус окрестности
minPts = 3   # Минимальное количество точек в окрестности

numIterations = 1  # количество замеров для усреднения


def distance(point1, point2):
    """Вычисление евклидова расстояния между двумя точками"""
    return np.sqrt(np.sum((point1 - point2) ** 2))


def find_neighbors(data, point_index):
    """Нахождение соседей для заданной точки"""
    neighbors = []  # Создаем пустой список для соседей
    # Проходим по каждой точке в данных
    for i in range(len(data)):
        # Проверяем, не является ли точка самой собой и не превышает ли расстояние до нее epsilon
        if i != point_index and distance(data[i], data[point_index]) <= epsilon:
            # Если условия выполняются, добавляем индекс точки в список соседей
            neighbors.append(i)
    return neighbors  # Возвращаем список соседей

def expand_cluster(data, start_point, cluster_id, visited, clusters):
    """Расширение кластера"""
    stack = [start_point]  # Стек для отслеживания точек, которые нужно обработать
    visited.add(start_point)  # Добавляем начальную точку в множество посещенных
    while stack:
        point_index = stack.pop()  # Извлекаем точку из стека
        clusters[cluster_id].append(point_index)  # Добавляем точку в текущий кластер
        neighbors = find_neighbors(data, point_index)  # Находим соседей для текущей точки
        for neighbor in neighbors:
            if neighbor not in visited:
                stack.append(neighbor)  # Добавляем соседа в стек
                visited.add(neighbor)  # Помечаем соседа как посещенного

def dbscan(data):
    """Реализация алгоритма DBSCAN"""
    clusters = {}  # Словарь для хранения кластеров
    cluster_id = 0  # Идентификатор текущего кластера
    visited = set()  # Множество посещенных точек
    for i, point in enumerate(data):
        if i not in visited:
            neighbors = find_neighbors(data, i)
            if len(neighbors) >= minPts:
                clusters[cluster_id] = []
                expand_cluster(data, i, cluster_id, visited, clusters)
                cluster_id += 1
    return clusters


def visualize_clusters(data, clusters, noise_points):
    """Визуализация данных и кластеров"""
    plt.figure(figsize=(10, 6))
    plt.scatter(data[:, 0], data[:, 1], c='black', marker='.', label='Шумовые точки')
    for cluster_id, cluster_points in clusters.items():
        cluster_points_data = data[cluster_points]
        plt.scatter(cluster_points_data[:, 0], cluster_points_data[:, 1], label=f'Кластер {cluster_id}', alpha=0.7, s=30)
    plt.title('DBSCAN, epsilon=1, minPts=3')
    plt.xlabel('X')
    plt.ylabel('Y')
    legend = plt.legend()
    legend.get_frame().set_alpha(1.0)
    plt.grid(True)
    plt.show()


# Считывание данных
dataPoints = np.loadtxt('my_mass10000.txt', dtype=float)



#  Замеры времени
print("Замер времени работы алгоритма DBSCAN")
print(time.ctime())

total_execution_time = 0
for i in range(numIterations):
    print("Замер {}".format(i))
    start_time = time.time()  # начальная точка замера времени

    # измеряемый код
    clusters = dbscan(dataPoints)

    end_time = time.time()  # конечная точка замера времени
    execution_time = end_time - start_time  # вычисление времени работы программы
    total_execution_time += execution_time

average_execution_time = (total_execution_time / numIterations) * 1000

# вывод времени работы программы в секундах
print("Результат для массива (10000, 3), Python, CPU: {} миллисекунд".format(average_execution_time))

