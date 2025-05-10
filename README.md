# Easy-Tensor-Compiler
## Запуск проекта
Запуск через CMake
```
mkdir build

cd build
```

Далее выбираем с какими настройками собирать проект.

Есть два варианта ```Debug``` и ```Release```:


```
cmake -DCMAKE_BUILD_TYPE=Debug ..

make
```

или

```
cmake -DCMAKE_BUILD_TYPE=Release ..

make
```

В результате будет создано два исполняемых файла: TESTS, MAIN

Первый позволяет запустить тесты проекта, второй - просто запускает код в файле ```Main.cc```
