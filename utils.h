#ifndef UTILS_H
#define UTILS_H


namespace SYS {

//преобразование типа перечисления с ограниченной областью видимости
//функция дает результат во время компиляции (constexpr), применяет автоматический вывод типа (auto, верно для C++14),
//не возвращает исключений (noexcept)
template<typename T>
constexpr auto toUType(T element) noexcept
{
    return static_cast<std::underlying_type_t<T>>(element);
}

}



#endif // UTILS_H
