#include <iostream>
#include <random>
#include <fstream>
#include <sstream>
#include <cmath>
#include <limits>
#include <chrono>
#include <omp.h>


using namespace std;


// Размерности массивов
const int size_mass1 = 10000;
const int size_mass2 = 100000;
const int size_mass3 = 1000000;

const int dim = 3; // размерность пространства 
const int k = 3; // количество кластеров

const int m = 2; // Экспоненциальный вес
const int max_iterations = 10; // Максимальное количество итераций
const float epsilon = 0.0001; // Пороговое значение для остановки

const int numIterations = 1; // количество замеров времени




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



void startingCenters(float** centers, float** data) {

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
void updateMembership(float** dataPoints, int size_mass, float** centers, float** distances, float** membership) {

    // Вычисление евклидовых расстояний от каждой точки до каждого центра

    // тут что-то надо придумать
    #pragma omp parallel for
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

    //тут что-то нужно придумать
    #pragma omp parallel for collapse(2)
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
void updateCenters(float** data, int mass_size, float** centers, float** membership) {


    // пересчёт центров
    for (int num_center = 0; num_center < k; num_center++) {


        // возведение элементов матрицы в степень
#pragma omp parallel num_threads(8)
        {
#pragma omp for schedule(static, 10)
            for (int i = 0; i < mass_size; i++) {
                for (int j = 0; j < k; j++) {
                    membership[i][j] = pow(membership[i][j], m);
                }
            }
        }

        float numerator[dim] = { 0.0f };
        float denominator = 0.0f;

        // вычисление числителя и знаменателя 

        // тут что-то надо придумать
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
bool converged(float** new_centers, float** old_centers) {
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
void c_means(float** data, int mass_size, float** centers, float** oldCenters, float** membership, float** distances) {

    // Инициализация начальных центроид
    startingCenters(centers, data);

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
        updateMembership(data, mass_size, centers, distances, membership);

        // Пересчет центров
        updateCenters(data, mass_size, centers, membership);

        // Условие остановки алгоритма
//        if (converged(centers, oldCenters, epsilon, k)) {
//            cout << "The algorithm converged on iteration " << iteration << endl;
//            break;
//        }

        // Обновление центров
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

    // Матрицы степеней принадлежности
    float** membership1 = new float* [size_mass1];
    for (int i = 0; i < size_mass1; i++) {
        membership1[i] = new float[k];
    }

    float** membership2 = new float* [size_mass2];
    for (int i = 0; i < size_mass2; i++) {
        membership2[i] = new float[k];
    }

    float** membership3 = new float* [size_mass3];
    for (int i = 0; i < size_mass3; i++) {
        membership3[i] = new float[k];
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




    // Считывание данных (10000, 3) из файла и запись в массив
    readFromFile(myArray1, size_mass1, file1);

    // Считывание данных (100000, 3) из файла и запись в массив
    readFromFile(myArray2, size_mass2, file2);

    // Считывание данных (1000000, 3) из файла и запись в массив
    for (int i = 0; i <= 9; i++) {
        readFromFile(myArray3 + i * 100000, size_mass3 / 10, i);
    }



    cout << "C-means algorithm. Start of program timing." << endl << endl;

    //cout << "Start timing mass (10000,3)" << endl << endl;

    // Замер времени для массива размерностью (10000, 3)
    double total_duration1 = 0.0f;
    for (int i = 0; i < numIterations; i++) {
        //cout << "timing iteration: " << i << endl << endl;
        auto start1 = chrono::steady_clock::now();
        c_means(myArray1, size_mass1, centers1, oldCenters1, membership1, distances1);
        auto end1 = chrono::steady_clock::now();
        auto duration1 = chrono::duration_cast<chrono::milliseconds>(end1 - start1).count();
        total_duration1 += duration1;
    }
    //среднее время из 10 замеров
    double average_duration1 = total_duration1 / numIterations;
    //cout << "CPU C++ (10000,3): " << average_duration1 << " ms." << endl << endl;

    //cout << "Start timing mass (100000,3)" << endl << endl;

    // Замер времени для массива размерностью (100000, 3)
    double total_duration2 = 0.0f;
    for (int i = 0; i < numIterations; i++) {
        //cout << "timing iteration: " << i << endl << endl;
        auto start2 = chrono::steady_clock::now();
        c_means(myArray2, size_mass2, centers2, oldCenters2, membership2, distances2);
        auto end2 = chrono::steady_clock::now();
        auto duration2 = chrono::duration_cast<chrono::milliseconds>(end2 - start2).count();
        total_duration2 += duration2;
    }
    //среднее время из 10 замеров
    double average_duration2 = total_duration2 / numIterations;
    //cout << "CPU C++ (100000,3): " << average_duration2 << " ms." << endl << endl;

    //cout << "Start timing mass (1000000,3)" << endl << endl;

    // Замер времени для массива размерностью (1000000, 3)
    double total_duration3 = 0.0f;
    for (int i = 0; i < numIterations; i++) {
        cout << "timing iteration: " << i << endl << endl;

        auto start3 = chrono::steady_clock::now();
        c_means(myArray3, size_mass3, centers3, oldCenters3, membership3, distances3);
        auto end3 = chrono::steady_clock::now();
        auto duration3 = chrono::duration_cast<chrono::milliseconds>(end3 - start3).count();
        total_duration3 += duration3;
    }
    //среднее время из 10 замеров
    double average_duration3 = total_duration3 / numIterations;
    //cout << "CPU C++ (1000000,3): " << average_duration3 << " ms." << endl << endl;


    // Итоговый вывод результатов
    cout << endl << endl << "Results:" << endl << endl;
    cout << "C-means algorithm. Program running time averaged over " << numIterations << " iterations." << endl << endl;

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

    // удаление массивов для хранения координат центров
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



    // удаление массивов для матриц степеней принадлежности
    for (int i = 0; i < size_mass1; i++) {
        delete[] membership1[i];
    }
    delete[] membership1;

    for (int i = 0; i < size_mass2; i++) {
        delete[] membership2[i];
    }
    delete[] membership2;

    for (int i = 0; i < size_mass3; i++) {
        delete[] membership3[i];
    }
    delete[] membership3;


    // удаление массивов для вычисления евклидовых расстояний
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


    return 0;
}