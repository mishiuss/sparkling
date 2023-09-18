## Модули

Фреймворк *Sparkling* состоит из двух частей:

* scala модуль **heaven**, содержащий реализации алгоритмов и мер качества. Они написаны на Scala 2.11.12 и 
не требуют дополнительных зависимостей, помимо Apache Spark. Отметим, что модуль уже скомпилирован 
(файл [heaven.jar](../bin/heaven.jar)), так что необходимо только указать в приложении Spark на .jar файл;

* python модуль **sparkling**, отвечающий за [препроцессинг](GLOSSARY_RU.md#препроцессинг), 
[процесс оптимизации](GLOSSARY_RU.md#процесс-оптимизации) и взаимодействие с пользователем.
Этому модулю необходимы некоторые [модули python](MODULES_RU.md#python-зависимости).
Для управления зависимостями рекомендуется использовать [conda](https://docs.conda.io/en/latest/).

### Python зависимости

Список зависимостей находится в папке [requirements](../requirements). Большинство зависимостей фреймворка *Sparkling* 
используется только в случае необходимости, а потому пользователь может установить только то, что ему требуется:

* [minimal.txt](../requirements/minimal.txt) - обязательные зависимости, достаточны для обработки табличных данных;
* [meta.txt](../requirements/meta.txt) - зависимости, необходимые для [рекомендации меры](GLOSSARY_RU.md#алгоритм-рекомендации);
* [deep.txt](../requirements/deep.txt) - основные зависимости для обработки модальностей с текстами и изображениями;
* [pytorch.txt](../requirements/pytorch.txt) - использовать [pytorch](https://pytorch.org/) как фреймворк глубокого обучения;
* [tensorflow.txt](../requirements/tensorflow.txt) - использовать [tensorflow](https://www.tensorflow.org/) как фреймворк глубокого обучения.

Мы рекомендуем установить [pytorch.txt](../requirements/pytorch.txt), чтобы был доступен весь функционал *Sparkling*.