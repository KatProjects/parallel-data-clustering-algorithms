import numpy as np
from sklearn.cluster import DBSCAN
import time

# количество итераций для замера времени
numIterations = 1

# считывание данных из файла в массив

data = np.loadtxt('my_mass100000.txt', dtype=float)

#  Замеры времени
print("Замер времени работы алгоритма DBSCAN на 10 итерациях, массив (100000, 3).")
print(time.ctime())

total_execution_time = 0
for i in range(numIterations):
    start_time = time.time()  # начальная точка замера времени
    # измеряемый код

    # Создание объекта DBSCAN с указанием параметров
    clustering = DBSCAN(eps=1.43, min_samples=3).fit(data)
    labels = clustering.labels_

    end_time = time.time()  # конечная точка замера времени
    execution_time = end_time - start_time  # вычисление времени работы программы
    total_execution_time += execution_time

average_execution_time = (total_execution_time / numIterations) * 1000

# вывод времени работы программы в секундах
print("Результат для массива (100000, 3), sklearn DBSCAN: {} миллисекунд".format(average_execution_time))
