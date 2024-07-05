import numpy as np
from sklearn.cluster import KMeans
import time  # импорт библиотеки для замера времени

# количество итераций для замера времени
numIterations = 100

# считывание данных из файла в массив

data = np.loadtxt('my_mass100.txt', dtype=float)

#  Замеры времени
print("Замер точности работы алгоритма K-means++ scikit-learn")
print(time.ctime())


def eucl(centers, real_centers, eps, k=3):
    print("Установлено реальное количество кластеров n_clusters=3, эпсилон {}".format(eps))
    for i in range(k):
        min_dist = float("inf")
        for j in range(len(real_centers)):
            dist = np.linalg.norm(centers[i] - real_centers[j])
            min_dist = min(min_dist, dist)
        print("Центр {}, евкл: {}".format(centers[i], min_dist))


realcenters = np.array([[0, 0, 0], [3, 3, 3], [-3, -3, -3]])

# Создание объекта CMeans с указанием числа кластеров
k_means100 = KMeans(n_clusters=3, tol=1e-5)
k_means100.fit(data)
eucl(k_means100.cluster_centers_, realcenters, 0.00001)


