#ifndef DEVICE_HPP
#define DEVICE_HPP

#include <type_traits>

namespace Device {

struct CPU {};
struct GPU {};

template <typename T>
struct IsTag {
	static constexpr bool value = std::is_same<T, CPU>::value || std::is_same<T, GPU>::value;
};

template <typename T>
constexpr bool IsTag_v = IsTag<T>::value;

}

#endif // UTIL_HPP