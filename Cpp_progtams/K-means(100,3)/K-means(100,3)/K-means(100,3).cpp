#include <iostream>
#include <random>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <chrono>


using namespace std;


const int size_mass = 100;   // Размерность массива 
const int dim = 3;           // размерность пространства 
int k = 3;                   // количество кластеров (может уменьшиться в процессе)



// Путь к файлу с данными
string file = "C:\\Users\\Катя\\Desktop\\Mass_100,10000,100000\\my_mass100.txt";



// Функция для считывания из файла данных с размерностью (100, 3)
void readFromFile(float** data, int size, string fileName) {
    string filename = fileName;
    ifstream file(filename);
    if (file.is_open()) {
        string line;
        int i = 0;
        while (getline(file, line) && i < size) {
            istringstream iss(line);
            float x, y, z;
            if (iss >> x >> y >> z) {
                data[i][0] = x;
                data[i][1] = y;
                data[i][2] = z;
                i++;
            }
        }
        file.close();
    }
    else {
        cerr << "Error opening file: " << filename << endl;
    }
}




void startingCenters(float** centers, float** data, int k) {

    float minValues[dim] = { 0.0f };
    float maxValues[dim] = { 0.0f };

    for (int i = 0; i < dim; i++) {
        minValues[i] = numeric_limits<float>::max();
        maxValues[i] = numeric_limits<float>::lowest();
    }

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < dim; j++) {
            centers[i][j] = 0;
        }
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < k; j++) {
            minValues[i] = min(minValues[i], data[j][i]);
            maxValues[i] = max(maxValues[i], data[j][i]);
        }
    }

    random_device rd;
    mt19937 gen(rd());

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < dim; j++) {
            uniform_real_distribution<> dis(minValues[j], maxValues[j]);
            centers[i][j] = dis(gen);
        }
    }
}





// Функция для заполнения кластеров: точки, центры, кол-во точек, двум масс расстояний, массив индексов кластеров
void clusterDistribution(float** data, float** centers, int size_mass, float** distances, int* cluster_indexes, int k) {

    // Вычисление евклидовых расстояний от каждой точки до каждого центра
    for (int point = 0; point < size_mass; point++) {
        for (int num_center = 0; num_center < k; num_center++) {
            float sum = 0.0f;
            for (int coord = 0; coord < dim; coord++) {
                sum += pow((data[point][coord] - centers[num_center][coord]), 2);
            }
            float dist = sqrt(sum);
            distances[point][num_center] = dist;
        }
    }

    // Индекс минимального расстояния - это индекс кластера, к которому определяется точка
    for (int point = 0; point < size_mass; point++) {
        int min_index = 0;
        float min_distance = distances[point][0];  // предполагается, что первое расстояние минимальное


        // сравнение дистанций, нахождение минимальной
        for (int dist_id = 1; dist_id < k; dist_id++) {
            if (distances[point][dist_id] < min_distance) {
                min_distance = distances[point][dist_id];
                min_index = dist_id;
            }
        }

        // сохраняется индекс центра с минимальным расстояниянием для текущей точки
        cluster_indexes[point] = min_index;

        // cluster_indexes содержит номера кластеров для индексов каждой точки
    }
}




// Функция для пересчёта центров кластеров
void recalculationCenters(float** data, int size_mass, int* cluster_indexes, float** centers, int k) {

    // массив для подсчёта входящих индексов
    int* count_cluster_i = new int[k]();

    for (int i = 0; i < size_mass; i++) {
        int cluster_id = cluster_indexes[i];
        count_cluster_i[cluster_id]++;
    }

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < dim; j++) {
            float sum = 0.0f;
            for (int point = 0; point < size_mass; point++) {
                if (cluster_indexes[point] == i) {
                    sum += data[point][j];
                }
            }
            if (count_cluster_i[i] != 0) {
                centers[i][j] = sum / count_cluster_i[i];
            }
            else {
                k = k - 1;
                continue;
            }
        }
    }

    delete[] count_cluster_i;
}



// Функция для проверки сходимости алгоритма 
bool converged(float** new_centers, float** old_centers, float epsilon, int k) {
    // Проверка сходимости по каждой компоненте центров
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < dim; j++) {
            // Если разница между новым и предыдущим центром больше epsilon, алгоритм не сошелся
            if (fabs(new_centers[i][j] - old_centers[i][j]) >= epsilon) {
                return false;
            }
        }
    }
    return true; // Если ни одна разница не превышает epsilon, алгоритм считается сойденным
}



void k_means(float** data, int size_mass, int* cluster_indexes, float** centers, float** oldCenters, float** distances, int max_iterations, float epsilon) {


    // Инициализация начальных центров кластеров
    startingCenters(centers, data, k);


    // Начальные центры
    cout << "Start centers: " << endl;
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < dim; j++) {
            cout << centers[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;


    // Копирование текущих центров в массив oldCenters
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < dim; j++) {
            oldCenters[i][j] = centers[i][j];
        }
    }

    int iteration = 0;
    while (iteration < max_iterations) {

        // Заполнение кластеров
        clusterDistribution(data, centers, size_mass, distances, cluster_indexes, k);

        // Пересчёт новых центров кластеров
        recalculationCenters(data, size_mass, cluster_indexes, centers, k);


        // Условие остановки алгоритма
        if (iteration > 0 && converged(centers, oldCenters, epsilon, k)) {
            cout << "The algorithm converged on iteration " << iteration << "." << endl << endl;
            break;
        }

        // Обновляем oldCenters
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < dim; j++) {
                oldCenters[i][j] = centers[i][j];
            }
        }

        iteration++;
    }

    cout << "Result centers:" << endl;
    // Вывод центров
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < dim; j++) {
            cout << centers[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;


}


int main() {

    const int max_iterations = 100;        // Максимальное количество итераций
    const float epsilon = 0.000001;        // Пороговое значение для остановки



    // Данные с точками
    float** myArray = new float* [size_mass];
    for (int i = 0; i < size_mass; i++) {
        myArray[i] = new float[dim];
    }

    // Центры
    float** centers = new float* [k];
    for (int i = 0; i < k; i++) {
        centers[i] = new float[dim];
    }

    // Старые центры
    float** oldCenters = new float* [k];
    for (int i = 0; i < k; i++) {
        oldCenters[i] = new float[dim];
    }

    // Евклидовы расстояния для точек до центров
    float** distances = new float* [size_mass];
    for (int i = 0; i < size_mass; i++) {
        distances[i] = new float[k];
    }

    // Массив для хранения индексов кластеров
    int* cluster_indexes = new int[size_mass];



    // Считывание данных из файла и запись в массив
    readFromFile(myArray, size_mass, file);

    // Вывод считаннных данных для проверки
 //   cout << "Data: " << endl;
 //   for (int i = 0; i < size_mass; i++) {
 //       for (int j = 0; j < dim; j++) {
 //           cout << myArray[i][j] << " ";
 //       }
 //       cout << endl;
 //   }
 //   cout << endl;



    cout << "K-means algorithm. Mass (100,3)." << endl << endl;
    k_means(myArray, size_mass, cluster_indexes, centers, oldCenters, distances, max_iterations, epsilon);



    // Освобождение памяти

    // Удаление точек данных
    for (int i = 0; i < size_mass; i++) {
        delete[] myArray[i];
    }
    delete[] myArray;


    // Удаление центров
    for (int i = 0; i < k; i++) {
        delete[] centers[i];
    }
    delete[] centers;

    for (int i = 0; i < k; i++) {
        delete[] oldCenters[i];
    }
    delete[] oldCenters;

    for (int i = 0; i < k; i++) {
        delete[] distances[i];
    }
    delete[] distances;

    // Удаление массива индексов кластеров
    delete[] cluster_indexes;


    return 0;
}
