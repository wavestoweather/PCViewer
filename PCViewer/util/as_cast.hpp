#pragma once

template<typename T, typename U>
inline constexpr T as(U&& in){
    if constexpr (std::is_same_v<T, U>)
        return in;
    else if constexpr (std::is_base_of_v<T, U> || std::is_base_of_v<U, T>)
        return dynamic_cast<T>(in);
    else if constexpr (std::is_pointer_v<T> && std::is_pointer_v<U>)
        return reinterpret_cast<T>(in);
    else if constexpr (std::is_reference_v<T> && std::is_reference_v<U>)
        return reinterpret_cast<T>(in);
    else
        return static_cast<T>(in);
}