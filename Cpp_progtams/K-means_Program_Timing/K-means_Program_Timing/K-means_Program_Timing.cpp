#include <iostream>
#include <random>
#include <fstream>
#include <sstream>
#include <cmath>
#include <limits>
#include <chrono>

using namespace std;

// Параметры

// Размерности массивов
const int size_mass1 = 10000;
const int size_mass2 = 100000;
const int size_mass3 = 1000000;

const int dim = 3; // размерность пространства 
const int k = 3; // количество кластеров


const int max_iterations = 10;        // Максимальное количество итераций
const float epsilon = 0.0001;        // Пороговое значение для остановки

const int numIterations = 100;          // количество замеров времени


// Пути к файлам с данными
string file1 = "C:\\Users\\Катя\\Desktop\\Mass_100,10000,100000\\my_mass10000.txt";     // для массива (10000, 3)
string file2 = "C:\\Users\\Катя\\Desktop\\Mass_100,10000,100000\\my_mass100000.txt";    // для массива (100000, 3)


// Функция для считывания из файлов данных с размерностями (10000, 3) и (100000, 3)
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


// Функция для считывания из файлов данных с размерностью (1000000, 3)
void readFromFile(float** data, int size, int part_number) {
    string filename = "C:\\Users\\Катя\\Desktop\\Mass_1000000\\my_mass1000000_part_" + to_string(part_number) + ".txt";
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
 //       if (iteration > 0 && converged(centers, oldCenters, epsilon, k)) {
 //           cout << "The algorithm converged on iteration " << iteration << "." << endl << endl;
 //           break;
 //       }

        // Обновляем oldCenters
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < dim; j++) {
                oldCenters[i][j] = centers[i][j];
            }
        }

        iteration++;
    }

}


int main() {


    // Выделение памяти под массивы

    // Данные с точками
    float** myArray1 = new float* [size_mass1];
    for (int i = 0; i < size_mass1; i++) {
        myArray1[i] = new float[dim];
    }

    float** myArray2 = new float* [size_mass2];
    for (int i = 0; i < size_mass2; i++) {
        myArray2[i] = new float[dim];
    }

    float** myArray3 = new float* [size_mass3];
    for (int i = 0; i < size_mass3; i++) {
        myArray3[i] = new float[dim];
    }

    // Центры
    float** centers1 = new float* [k];
    for (int i = 0; i < k; i++) {
        centers1[i] = new float[dim];
    }

    float** centers2 = new float* [k];
    for (int i = 0; i < k; i++) {
        centers2[i] = new float[dim];
    }
    float** centers3 = new float* [k];
    for (int i = 0; i < k; i++) {
        centers3[i] = new float[dim];
    }


    // Старые центры
    float** oldCenters1 = new float* [k];
    for (int i = 0; i < k; i++) {
        oldCenters1[i] = new float[dim];
    }

    float** oldCenters2 = new float* [k];
    for (int i = 0; i < k; i++) {
        oldCenters2[i] = new float[dim];
    }

    float** oldCenters3 = new float* [k];
    for (int i = 0; i < k; i++) {
        oldCenters3[i] = new float[dim];
    }



    // Для вычисления евклидовых расстояний
    float** distances1 = new float* [size_mass1];
    for (int i = 0; i < size_mass1; i++) {
        distances1[i] = new float[k];
    }

    float** distances2 = new float* [size_mass2];
    for (int i = 0; i < size_mass2; i++) {
        distances2[i] = new float[k];
    }

    float** distances3 = new float* [size_mass3];
    for (int i = 0; i < size_mass3; i++) {
        distances3[i] = new float[k];
    }


    // Массивы для хранения индексов кластеров
    int* cluster_indexes1 = new int[size_mass1];
    int* cluster_indexes2 = new int[size_mass2];
    int* cluster_indexes3 = new int[size_mass3];


    // Считывание данных (10000, 3) из файла и запись в массив
    readFromFile(myArray1, size_mass1, file1);

    // Считывание данных (100000, 3) из файла и запись в массив
    readFromFile(myArray2, size_mass2, file2);

    // Считывание данных (1000000, 3) из файла и запись в массив
    for (int i = 0; i <= 9; i++) {
        readFromFile(myArray3 + i * 100000, size_mass3 / 10, i);
    }


    //cout << "K-means algorithm. Start of program timing." << endl << endl;


    // Замер времени для массива размерностью (10000, 3)
    //cout << "Start timing mass (10000,3)" << endl << endl;
    double total_duration1 = 0.0f;
    for (int i = 0; i < numIterations; i++) {
    //    cout << "timing iteration: " << i << endl << endl;
        auto start1 = chrono::steady_clock::now();
        k_means(myArray1, size_mass1, cluster_indexes1, centers1, oldCenters1, distances1, max_iterations, epsilon);
        auto end1 = chrono::steady_clock::now();
        auto duration1 = chrono::duration_cast<chrono::milliseconds>(end1 - start1).count();
        total_duration1 += duration1;
    }
    //среднее время из 10 замеров
    double average_duration1 = total_duration1 / numIterations;
    //cout << "CPU C++ (10000,3): " << average_duration1 << " ms." << endl << endl;



    // Замер времени для массива размерностью (100000, 3)
    //cout << "Start timing mass (100000,3)" << endl << endl;
    double total_duration2 = 0.0f;
    for (int i = 0; i < numIterations; i++) {
    //    cout << "timing iteration: " << i << endl << endl;
        auto start2 = chrono::steady_clock::now();
        k_means(myArray2, size_mass2, cluster_indexes2, centers2, oldCenters2, distances2, max_iterations, epsilon);
        auto end2 = chrono::steady_clock::now();
        auto duration2 = chrono::duration_cast<chrono::milliseconds>(end2 - start2).count();
        total_duration2 += duration2;
    }
    //среднее время из 10 замеров
    double average_duration2 = total_duration2 / numIterations;
    //cout << "CPU C++ (100000,3): " << average_duration2 << " ms." << endl << endl;



    // Замер времени для массива размерностью (1000000, 3)
    //cout << "Start timing mass (1000000,3)" << endl << endl;
    double total_duration3 = 0.0f;
    for (int i = 0; i < numIterations; i++) {
    //    cout << "timing iteration: " << i << endl << endl;

        auto start3 = chrono::steady_clock::now();
        k_means(myArray3, size_mass3, cluster_indexes3, centers3, oldCenters3, distances3, max_iterations, epsilon);
        auto end3 = chrono::steady_clock::now();
        auto duration3 = chrono::duration_cast<chrono::milliseconds>(end3 - start3).count();
        total_duration3 += duration3;
    }
    //среднее время из 10 замеров
    double average_duration3 = total_duration3 / numIterations;
    //cout << "CPU C++ (1000000,3): " << average_duration3 << " ms." << endl << endl;


    // Итоговый вывод результатов
    cout << endl << endl << "Results:" << endl << endl;
    cout << "K-means algorithm. Program running time averaged over " << numIterations << " iterations." << endl << endl;

    cout << "CPU C++ (10000,3): " << average_duration1 << " ms." << endl;
    cout << "CPU C++ (100000,3): " << average_duration2 << " ms." << endl;
    cout << "CPU C++ (1000000,3): " << average_duration3 << " ms." << endl;



    // Освобождение памяти

    // удаление массивов для хранения данных с точками
    for (int i = 0; i < size_mass1; i++) {
        delete[] myArray1[i];
    }
    delete[] myArray1;

    for (int i = 0; i < size_mass2; i++) {
        delete[] myArray2[i];
    }
    delete[] myArray2;

    for (int i = 0; i < size_mass3; i++) {
        delete[] myArray3[i];
    }
    delete[] myArray3;


    // Удаление центров
    for (int i = 0; i < k; i++) {
        delete[] centers1[i];
    }
    delete[] centers1;

    for (int i = 0; i < k; i++) {
        delete[] centers2[i];
    }
    delete[] centers2;

    for (int i = 0; i < k; i++) {
        delete[] centers3[i];
    }
    delete[] centers3;


    for (int i = 0; i < k; i++) {
        delete[] oldCenters1[i];
    }
    delete[] oldCenters1;

    for (int i = 0; i < k; i++) {
        delete[] oldCenters2[i];
    }
    delete[] oldCenters2;

    for (int i = 0; i < k; i++) {
        delete[] oldCenters3[i];
    }
    delete[] oldCenters3;


    // Удаление массивов для вычисления евклидовых расстояний
    for (int i = 0; i < k; i++) {
        delete[] distances1[i];
    }
    delete[] distances1;

    for (int i = 0; i < k; i++) {
        delete[] distances2[i];
    }
    delete[] distances2;

    for (int i = 0; i < k; i++) {
        delete[] distances3[i];
    }
    delete[] distances3;



    // Удаление массива индексов кластеров
    delete[] cluster_indexes1;
    delete[] cluster_indexes2;
    delete[] cluster_indexes3;


    return 0;
}
