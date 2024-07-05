import numpy as np
import matplotlib.pyplot as plt

# Параметры:
k = 3


# функция для отображения точек в виде графика в исходном виде одним цветом:
def start_graph(data):
    # Определение размерности данных
    dimensions = data.shape[1]

    if dimensions == 1:
        x = data[:, 0]
        plt.hlines(0, xmin=min(x) - 1, xmax=max(x) + 1)  # Отрисовка оси X
        plt.scatter(x, [0] * len(x), s=30)  # Точки данных на оси X
        plt.title("Массив 100 точек в одномерном пространстве")
        plt.xlabel("Значение")
        plt.ylabel("Ось")
        plt.grid(True)
        plt.yticks([])
        plt.show()
    elif dimensions == 2:
        x = data[:, 0]
        y = data[:, 1]
        plt.scatter(x, y)
        plt.title("Массив 150 точек в двухмерном пространстве")
        plt.grid(True)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
    elif dimensions == 3:
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z)
        plt.grid(True)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title("Массив 100 точек в трёхмерном пространстве")
        plt.show()
    else:
        print("Массив не является одномерным, двумерным или трехмерным\n")


# график с разделением точек на кластеры по цветам
def plot_clusters(clusters, centers):
    # cmap = 'plasma'  # выбираем палитру цветов
    alpha_value = 0.7  # устанавливаем значение прозрачности
    colors = ['red', 'blue', 'lime', 'brown']

    if centers.shape[1] == 1:
        for i, cluster in enumerate(clusters):
            cluster_points = np.array(cluster)
            plt.scatter(cluster_points[:, 0], [0] * len(cluster_points), color=colors[i], label=f'Кластер {i+1}', s=30, alpha=alpha_value)
        plt.scatter(centers[:, 0], [0] * len(centers), color='black', marker='x', label='Центры', s=50, alpha=alpha_value)
        plt.xlabel("Значение")
        plt.ylabel("Ось")
        plt.title("K-means 1D")
        plt.legend()
        plt.grid(True)
        plt.show()
    elif centers.shape[1] == 2:
        for i, cluster in enumerate(clusters):
            cluster_points = np.array(cluster)
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=f'Cluster {i}', s=30, alpha=alpha_value)
        plt.scatter(centers[:, 0], centers[:, 1], color='black', marker='x', label='Centers', s=60, alpha=alpha_value)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title("K-means, Centers=3")
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.show()
    elif centers.shape[1] == 3:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        for i, cluster in enumerate(clusters):
            cluster_points = np.array(cluster)
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], color=colors[i], label=f'Cluster {i}', alpha=alpha_value)
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], color='black', marker='x', label='Centers', s=50, alpha=alpha_value)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title("K-means, centers=3")
        plt.legend()
        # plt.grid(True)
        plt.show()
    else:
        print("Массив не является одномерным, двумерным или трехмерным\n")
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


# коэффициент силуэта
def silhouette_score(clusters):
    silhouette_scores = []

    for cluster_idx, cluster in enumerate(clusters):
        for point in cluster:
            # Вычисляется a - среднее расстояние внутри кластера
            a = np.mean([np.linalg.norm(point - other_point) for other_point in cluster if not np.array_equal(point, other_point)])

            # Вычисляется b - среднее расстояние до точек ближайшего соседнего кластера
            b = float('inf')  # начальное значение - бесконечность
            for other_cluster_idx, other_cluster in enumerate(clusters):
                if cluster_idx != other_cluster_idx:
                    avg_distance = np.mean([np.linalg.norm(point - other_point) for other_point in other_cluster])
                    if avg_distance < b:
                        b = avg_distance

            # Вычисляется коэффициент силуэта для данной точки
            silhouette_coefficient = (b - a) / max(a, b)
            silhouette_scores.append(silhouette_coefficient)

    # Возвращается средний коэффициент силуэта для всех точек
    return np.mean(silhouette_scores)


# Работа алгоритма
def k_means(points_data, max_iterations=10, epsilon=1e-4):
    print("Работа алгоритма K-means\n")

    start_graph(points_data)

    # Просмотр переданных данных
    print("Данные: \n{}\n".format(points_data))

    # генерация случайных 3-х начальных центров
    centers = starting_centers(points_data, k)

    # просмотр сгенерированных начальных центров
    print("\nГенерация случайных центров:\n{}".format(centers))

    # Итерации алгоритма

    iteration = 0  # инициализация стартового значения
    clusters = [[]]
    # цикл
    while iteration < max_iterations:
        print("\nИтерация {}.\n".format(iteration + 1))

        # приближение данных к центрам, формирование кластеров
        clusters = cluster_distribution(points_data, centers)

        # вывод текущего приближения на итерации
        # plot_clusters(clusters, centers)

        # просмотр распределения точек по кластерам
        print("Кластеры:")
        for cluster_index in range(0, len(clusters)):
            print("Кластер {}: {}".format(cluster_index, clusters[cluster_index]))

        # пересчёт новых центров кластеров
        new_centers = recalculation_centers(clusters)

        # условие останова, если модуль всех расстояний меньше эпсилон - завершаем работу алгоритма
        try:
            if np.all(np.abs(new_centers - centers) < epsilon):
                print("Алгоритм сошёлся на итерации {}".format(iteration))
                break
        except ValueError as err:
            print("Размерность кластеров уменьшилась")

        centers = new_centers

        # просмотр пересчитанных центров
        print("\nПересчёт центров:\n{}".format(centers))

        # вывод графика по итерациям
        # plot_clusters(clusters, centers)

        iteration += 1

    # Построение результирующего графика
    plot_clusters(clusters, centers)
    score = silhouette_score(clusters)
    # Вывод оценки кластеризации с помощью коэффициента силуэта
    print("\nОценка кластеризации.\nКоэффициент силуэта: {}".format(score))
    if score > 0.7:
        print("Кластеризация хорошая.")
    elif (score > 0.5) and (score <= 0.7):
        print("Кластеризация приемлемая.")
    else:
        print("Недостаточная кластеризация")


data = np.loadtxt('my_mass100.txt', dtype=float)
k_means(data)

