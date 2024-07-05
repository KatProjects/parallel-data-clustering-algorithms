#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <stack>
#include <chrono>
#include <omp.h>


using namespace std;


// Параметры:
// 
// Размерности массивов
const int size_mass1 = 10000;
const int size_mass2 = 100000;
const int size_mass3 = 1000000;

// Размерность пространства
const int dim = 3;

// Параметры работы алгоритма
const float epsilon = 1.43;          // радиус окрестности
const int minPts = 3;                // минимальное количество соседей

// Количество итераций для замеров времени
const int numIterations = 1;



// Пути к файлам с данными
string file1 = "C:\\Users\\Катя\\Desktop\\Mass_100,10000,100000\\my_mass10000.txt";     // для массива (10000, 3)
string file2 = "C:\\Users\\Катя\\Desktop\\Mass_100,10000,100000\\my_mass100000.txt";    // для массива (100000, 3)


// Функция для считывания данных из файлов с размерностями (10000, 3) и (100000, 3)
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


// Функция для считывания данных из файла размерностью (1000000, 3)
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




// Вычисление евклидова расстояния
float distance(float** data, int point_index1, int point_index2) {
    float sum = 0;
    for (int coord = 0; coord < dim; coord++) {
        sum += (data[point_index1][coord] - data[point_index2][coord]) * (data[point_index1][coord] - data[point_index2][coord]);
    }
    float dist = sqrt(sum);

    return dist;
}



// функция для поиска соседей у точки, результат в виде 0 и 1 будет лежать в массиве neighbours
void find_neighbours(float** data, int mass_size, int point_index, int* neighbours) {
    #pragma omp parallel for
    for (int i = 0; i < mass_size; i++) {
        if (i != point_index) {
            if (distance(data, point_index, i) <= epsilon) {
                neighbours[i] = 1;
            }
            else {
                neighbours[i] = 0;
            }
        }
    }
}



// функция для расширения кластера
void expand_cluster(float** data, int size, int start_point, int cluster_id, int* labels, int* neighbours) {

    //создание стека для отслеживания точек, которые нужно обработать
    stack<int> stack;

    // В стек добавляется начальная точка
    stack.push(start_point);

    // Пока стек не пустой
    while (!stack.empty()) {

        // получение точки из стека
        int point = stack.top();

        // удаление точки из стека
        stack.pop();

        // назначение точке метки текущего кластера
        labels[point] = cluster_id;

        // Осуществляется поиск её соседей, будут в neighbours
        for (int i = 0; i < size; i++) {
            neighbours[i] = 0;
        }

        find_neighbours(data, size, point, neighbours);

        // Подсчёт соседей
        int count = 0;
        #pragma omp parallel for reduction(+:count)
        for (int i = 0; i < size; i++) {
            if (neighbours[i] == 1) {
                count++;
            }
        }

        if (count >= minPts) {
            for (int i = 0; i < size; i++) {
                if (neighbours[i] == 1) {

                        if (labels[i] == -1) {
                            stack.push(i);
                            labels[i] = cluster_id;
                        }
                        if (labels[i] == -2) {
                            labels[i] = cluster_id;
                        }
                }
            }
        }
    }
}



// Основная фнункция алгоритма
void dbscan(float** data, int size, int* labels, int* neighbours) {

    // Начальное значение индекса кластера
    int cluster_id = 0;

   // #pragma omp parallel for schedule(dynamic)
    for (int point_index = 0; point_index < size; point_index++) {

        // если точка не посещена
        if (labels[point_index] == -1) {

            // Осуществляется поиск её соседей, будут в neighbours
            for (int i = 0; i < size; i++) {
                neighbours[i] = 0;
            }

            find_neighbours(data, size, point_index, neighbours);

            // Подсчёт соседей
            int count = 0;
            #pragma omp parallel for reduction(+:count)
            for (int i = 0; i < size; i++) {
                if (neighbours[i] == 1) {
                    count++;
                }
            }

            // Если количество соседей меньше заданного минимума
            if (count < minPts) {

                // Точка помечается как шумовая (-2)
                labels[point_index] = -2;
                continue;
            }

            // Если больше или равно
            if (count >= minPts) {

                // Точка считается ядерной

                // Расширение кластера
               // #pragma omp critical
               // {

                    expand_cluster(data, size, point_index, cluster_id, labels, neighbours);

                    // После того, как кластер завершён, увеличивается индекс для текущего
                    cluster_id++;
              //  }
            }
        }
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

    //float** myArray3 = new float* [size_mass3];
    //for (int i = 0; i < size_mass3; i++) {
    //    myArray3[i] = new float[dim];
    //}

    // для меток точек, изначально не посещённые (-1)
    int* labels1 = new int[size_mass1];
    for (int i = 0; i < size_mass1; i++) {
        labels1[i] = -1;
    }

    int* labels2 = new int[size_mass2];
    for (int i = 0; i < size_mass2; i++) {
        labels2[i] = -1;
    }

    //int* labels3 = new int[size_mass3];
    //for (int i = 0; i < size_mass3; i++) {
    //    labels3[i] = -1;
    //}

    // для меток соседей (0 - не сосед, 1 - сосед)
    int* neighbours1 = new int[size_mass1];
    int* neighbours2 = new int[size_mass2];
    //int* neighbours3 = new int[size_mass3];




    // Считывание данных (10000, 3) из файла и запись в массив
    readFromFile(myArray1, size_mass1, file1);

    // Считывание данных (100000, 3) из файла и запись в массив
    readFromFile(myArray2, size_mass2, file2);

    // Считывание данных (1000000, 3) из файла и запись в массив
   // for (int i = 0; i <= 9; i++) {
   //     readFromFile(myArray3 + i * 100000, size_mass3 / 10, i);
   // }


    cout << "DBSCAN algorithm with OpenMP. Start of program timing." << endl << endl;


    cout << "Start timing mass (10000,3)" << endl << endl;

    // Замер времени для массива размерностью (10000, 3)
    double total_duration1 = 0.0f;
    for (int i = 0; i < numIterations; i++) {
        cout << "timing iteration: " << i << endl << endl;
        auto start1 = chrono::steady_clock::now();
        dbscan(myArray1, size_mass1, labels1, neighbours1);
        auto end1 = chrono::steady_clock::now();
        auto duration1 = chrono::duration_cast<chrono::milliseconds>(end1 - start1).count();
        total_duration1 += duration1;
    }
    //среднее время из 10 замеров
    double average_duration1 = total_duration1 / numIterations;
    cout << "CPU C++ (10000,3): " << average_duration1 << " ms." << endl << endl;

    cout << "Start timing mass (100000,3)" << endl << endl;

    // Замер времени для массива размерностью (100000, 3)
    double total_duration2 = 0.0f;
    for (int i = 0; i < numIterations; i++) {
        cout << "timing iteration: " << i << endl << endl;
        auto start2 = chrono::steady_clock::now();
        dbscan(myArray2, size_mass2, labels2, neighbours2);
        auto end2 = chrono::steady_clock::now();
        auto duration2 = chrono::duration_cast<chrono::milliseconds>(end2 - start2).count();
        total_duration2 += duration2;
    }
    //среднее время из 10 замеров
    double average_duration2 = total_duration2 / numIterations;
    cout << "CPU C++ (100000,3): " << average_duration2 << " ms." << endl << endl;

    //cout << "Start timing mass (1000000,3)" << endl << endl;

    // Замер времени для массива размерностью (1000000, 3)
    //double total_duration3 = 0.0f;
    //for (int i = 0; i < numIterations; i++) {
        //    cout << "timing iteration: " << i << endl << endl;

      //  auto start3 = chrono::steady_clock::now();
      //  dbscan(myArray3, size_mass3, labels3, neighbours3);
      //  auto end3 = chrono::steady_clock::now();
      //  auto duration3 = chrono::duration_cast<chrono::milliseconds>(end3 - start3).count();
      //  total_duration3 += duration3;
    //}
    //среднее время из 10 замеров
    //double average_duration3 = total_duration3 / numIterations;
    // cout << "CPU C++ (1000000,3): " << average_duration3 << " ms." << endl << endl;


     // Итоговый вывод результатов
     cout << endl << endl << "Results:" << endl << endl;
     cout << "DBSCAN algorithm with OpenMP. Program running time averaged over " << numIterations << " iterations." << endl << endl;

     cout << "CPU C++ (10000,3): " << average_duration1 << " ms." << endl;
     cout << "CPU C++ (100000,3): " << average_duration2 << " ms." << endl;
    //cout << "CPU C++ (1000000,3): " << average_duration3 << " ms." << endl;



    // Освобождение памяти

    // удаление массивов для хранения данных с точками
    for (int i = 0; i < size_mass1; ++i) {
        delete[] myArray1[i];
    }
    delete[] myArray1;

    for (int i = 0; i < size_mass2; ++i) {
        delete[] myArray2[i];
    }
    delete[] myArray2;

    //for (int i = 0; i < size_mass3; ++i) {
    //    delete[] myArray3[i];
    //}
    //delete[] myArray3;


    // для массива с метками точек
    delete[] labels1;
    delete[] labels2;
    //delete[] labels3;


    // для массива с метками соседей
    delete[] neighbours1;
    delete[] neighbours2;
   // delete[] neighbours3;

    return 0;
}