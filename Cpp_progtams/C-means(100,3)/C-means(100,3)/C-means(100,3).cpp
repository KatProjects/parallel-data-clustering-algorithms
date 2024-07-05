#include <iostream>
#include <random>
#include <fstream>
#include <sstream>
#include <cmath>
#include <limits>
#include <chrono>

using namespace std;


// Размерность массива
const int size_mass = 100;

const int dim = 3; // размерность пространства 
const int k = 3; // количество кластеров

// Путь к файлу с данными
string file = "C:\\Users\\Катя\\Desktop\\Mass_100,10000,100000\\my_mass100.txt";


// Функция для считывания данных из файла
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



//  пересчёт матрицы степеней принадлежности на основе вычисленных центроид
void updateMembership(float** dataPoints, int size_mass, float** centers, float** distances, float** membership, int k, int m) {

    // Вычисление евклидовых расстояний от каждой точки до каждого центра
    for (int point = 0; point < size_mass; point++) {
        for (int num_center = 0; num_center < k; num_center++) {
            float sum = 0.0f;
            for (int coord = 0; coord < dim; coord++) {
                sum += pow((dataPoints[point][coord] - centers[num_center][coord]), 2);
            }
            float dist = sqrt(sum);
            distances[point][num_center] = dist;
        }
    }

    // Вычисление степеней принадлежности для каждой точки
    for (int i = 0; i < size_mass; i++) {
        for (int j = 0; j < k; j++) {
            float sumDistances = 0.0f;
            for (int c = 0; c < k; c++) {
                float distancesRatio = distances[i][j] / distances[i][c];
                sumDistances += pow(distancesRatio, (2.0f / (m - 1)));
            }
            membership[i][j] = 1.0f / sumDistances;
        }
    }
}



// функция для пересчёта центров
void updateCenters(float** data, int mass_size, float** centers, float** membership, int m) {


    // пересчёт центров
    for (int num_center = 0; num_center < k; num_center++) {


        // возведение элементов матрицы в степень
        for (int i = 0; i < mass_size; i++) {
            for (int j = 0; j < k; j++) {
                membership[i][j] = pow(membership[i][j], m);
            }
        }

        float numerator[dim] = { 0.0f };
        float denominator = 0.0f;

        // вычисление числителя и знаменателя 
        for (int i = 0; i < mass_size; i++) {
            for (int j = 0; j < dim; j++) {
                numerator[j] += membership[i][num_center] * data[i][j];
            }
            denominator += membership[i][num_center];
        }

        // пересчёт центра
        for (int j = 0; j < dim; j++) {
            centers[num_center][j] = numerator[j] / denominator;
        }

    }

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



// Основная функция алгоритма C-Means
void c_means(float** data, int mass_size, float** centers, float** oldCenters, float** membership, float** distances, int max_iterations, int m, float epsilon) {

    // Инициализация начальных центроид
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

    // Итерационный процесс
    int iteration = 0;
    while (iteration < max_iterations) {

        // Рассчет матрицы принадлежности на основе текущих центров
        updateMembership(data, mass_size, centers, distances, membership, k, m);

        // Пересчет центров
        updateCenters(data, mass_size, centers, membership, m);

        // Условие остановки алгоритма
        if (converged(centers, oldCenters, epsilon, k)) {
            cout << "The algorithm converged on iteration " << iteration << "." << endl << endl;
            break;
        }

        // Обновление центров
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < dim; j++) {
                oldCenters[i][j] = centers[i][j];
            }
        }

        iteration++;
    }

    cout << "Result centers:" << endl;
    // Вывод центров
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < dim; ++j) {
            cout << centers[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;

}



int main() {

    const int m = 2; // Экспоненциальный вес
    const int max_iterations = 200; // Максимальное количество итераций
    const float epsilon = 0.0001; // Пороговое значение для остановки


    // Выделение памяти
    // 
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


    // Матрицы степеней принадлежности
    float** membership = new float* [size_mass];
    for (int i = 0; i < size_mass; i++) {
        membership[i] = new float[k];
    }



    // Для вычисления евклидовых расстояний
    float** distances = new float* [size_mass];
    for (int i = 0; i < size_mass; i++) {
        distances[i] = new float[k];
    }



    // Считывание данных (100, 3) из файла и запись в массив
    readFromFile(myArray, size_mass, file);

    //Вывод считанных данных для проверки
 //   cout << "Data: " << endl;
 //   for (int i = 0; i < size_mass; i++) {
 //       for (int j = 0; j < dim; j++) {
 //           cout << myArray[i][j] << " ";
 //       }
 //       cout << endl;
 //   }



    cout << endl << "C-means algorithm." << endl;
    cout << "Mass(" << size_mass << "," << dim << "), k=" << k << ", epsilon = " << epsilon <<", max_iteration = " << max_iterations << endl << endl;

   // Запуск алгоритма
    c_means(myArray, size_mass, centers, oldCenters, membership, distances, max_iterations, m, epsilon);


    // Освобождение памяти

    // удаление массивов для хранения данных с точками
    for (int i = 0; i < size_mass; ++i) {
        delete[] myArray[i];
    }
    delete[] myArray;


    // удаление массивов для хранения координат центров
    for (int i = 0; i < k; i++) {
        delete[] centers[i];
    }
    delete[] centers;


    for (int i = 0; i < k; i++) {
        delete[] oldCenters[i];
    }
    delete[] oldCenters;



    // удаление массивов для матриц степеней принадлежности
    for (int i = 0; i < size_mass; i++) {
        delete[] membership[i];
    }
    delete[] membership;


    // удаление массивов для вычисления евклидовых расстояний
    for (int i = 0; i < k; i++) {
        delete[] distances[i];
    }
    delete[] distances;


    return 0;
}