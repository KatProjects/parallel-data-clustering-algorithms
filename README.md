## ВКР: "Параллельные алгоритмы кластеризации данных"

Этот репозиторий содержит реализации алгоритмов кластеризации данных на Python и C++, которые использовались у меня в дипломной работе.
К программам, написанным на языке C++ применены технологии для организации параллельных вычислений и сделаны замеры времени работы.
Получены выводы, что собственные реализации алгоритмов могут иметь преимущество над использованием готовых из библиотек.

### Структура проекта

- Python_programs/: Коды на Python
- C++ programs/: Коды на C++
- C++ with OpenMP programs/: Коды на C++ с применением технологии OpenMP
- CUDA C++ programs/: Коды на CUDA C++
- Data/: Данные для тестирования алгоритмов


### Файлы с массивами данных:

Все файлы содержат числа с плавающей точкой, разделённые пробелом и переходом на следующую строку.
Каждая строка содержит координаты одной точки в пространстве. Формат данных TXT.

Данные для замеров времени и точности, представляют собой три сферических кластера с разбросом точек вокруг центров (0, 0, 0), (3, 3, 3), (-3, -3, -3):

- my_mass100.txt: Набор данных из 100 элементов.

- my_mass10000.txt: Набор данных из 10,000 элементов.

- my_mass100000.txt: Набор данных из 100,000 элементов.

- my_mass1000000_part_0.txt ... my_mass1000000_part_9.txt: Набор данных из 1,000,000 элементов, разделенный на 10 файлов.

Дополнительные файлы данных, которые использовались для визуализаций:

- my_diff_mass_400.txt: Набор данных из 400 элементов, три кластера произвольных форм и шумовые точки.

- my_rand_500.txt: Набор данных из 500 элементов, плоский кластер с равномерно и случайно разбросанными точками на ограниченном пространстве.

- my_rand_150.txt: Набор данных из 150 элементов, три сферических кластера с точками, случайно разбросанными по ст. норм. распр. вокруг центров (0, 0, 0), (3, 3, 3), (-3, -3, -3).

