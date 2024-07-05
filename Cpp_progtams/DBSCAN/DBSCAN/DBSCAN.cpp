#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <stack>


using namespace std;


// Параметры:
const int size_mass = 100;
const int dim = 3;
const float epsilon = 1.43;
const int minPts = 3;


// Путь к файлу с данными
string file = "C:\\Users\\Катя\\Desktop\\Mass_100,10000,100000\\my_mass100.txt";



// Функция считывания с файла
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
int dbscan(float** data, int size, int* labels, int* neighbours) {

    // Начальное значение индекса кластера
    int cluster_id = 0;


    for (int point_index = 0; point_index < size; point_index++) {

        // если точка не посещена
        if (labels[point_index] == -1) {

            // Осуществляется поиск её соседей, будут в neighbours

            for (int i = 0; i < size_mass; i++) {
                neighbours[i] = 0;
            }

            find_neighbours(data, size, point_index, neighbours);

            // Подсчёт соседей
            int count = 0;
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

                expand_cluster(data, size, point_index, cluster_id, labels, neighbours);

                // После того, как кластер завершён, увеличивается индекс для текущего
                cluster_id++;
            }
        }
    }

    return cluster_id;
}



int main() {

    // Выделение памяти

    // для данных с точками
    float** myArray = new float* [size_mass];
    for (int i = 0; i < size_mass; i++) {
        myArray[i] = new float[dim];
    }

    // для меток точек, изначально не посещённые (-1)
    int* labels = new int[size_mass];
    for (int i = 0; i < size_mass; i++) {
        labels[i] = -1;
    }

    // для меток соседей (0 - не сосед, 1 - сосед)
    int* neighbours = new int[size_mass];



    // чтение данных их файла и запись в массив
    readFromFile(myArray, size_mass, file);



    // работа алгоритма DBSCAN,  вернёт количество кластеров
    int count_clusters = dbscan(myArray, size_mass, labels, neighbours);



    // Вывод меток
    cout << endl << "Labels: " << endl;
    for (int i = 0; i < size_mass; i++) {
        cout << labels[i] << " ";
    }



    // Группировка меток по кластерам
    cout << endl << "Clusters." << endl;
    for (int cluster_id = 0; cluster_id < count_clusters; cluster_id++) {
        cout << cluster_id << ": ";
        for (int point_index = 0; point_index < size_mass; point_index++) {
            if (labels[point_index] == cluster_id) {
                cout << point_index << " ";
            }
        }
        cout << endl;
    }

    cout << "Noise: ";
    for (int point_index = 0; point_index < size_mass; point_index++) {
        if (labels[point_index] == -2) {
            cout << point_index << " ";
        }
    }
    cout << endl << endl;



    // Освобождение памяти

    // для массива данных
    for (int i = 0; i < size_mass; i++) {
        delete[] myArray[i];
    }
    delete[] myArray;

    // для массива с метками точек
    delete[] labels;

    // для массива с метками соседей
    delete[] neighbours;


    return 0;
}