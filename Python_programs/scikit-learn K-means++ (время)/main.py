
import numpy as np
from sklearn.cluster import KMeans
import time
# количество итераций для замера времени
numIterations = 100

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

    # Создание объекта CMeans с указанием числа кластеров
    k_means = KMeans(n_clusters=3, max_iter=10, tol=1e-100)

    # Обучение модели на данных
    k_means.fit(data)

    # метки кластеров
    labels = k_means.labels_

    end_time = time.time()  # конечная точка замера времени
    execution_time = end_time - start_time  # вычисление времени работы программы
    total_execution_time += execution_time

average_execution_time = total_execution_time / numIterations

# вывод времени работы программы в секундах
print("Результат для массива (100, 3), sklearn KMeans++: {} секунд".format(average_execution_time))
