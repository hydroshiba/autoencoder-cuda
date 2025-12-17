#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <iostream>
#include <string_view>
// #include <stringstream>

#if defined(_MSC_VER)
	#define FUNC_SIG __FUNCSIG__
#else
	#define FUNC_SIG __PRETTY_FUNCTION__
#endif

// The macro users actually call
#define LOG(...) Logger::log(FUNC_SIG, __VA_ARGS__)

class Logger {
public:
	Logger() = delete;

	template <typename... Args>
	static void log(
		std::string_view signature,
		std::ostream& stream,
		Args&&... args
	) {
		std::string_view class_name = extract_class(signature);
		stream << "[" << class_name << "] ";
		((stream << std::forward<Args>(args) << ' '), ...);
		stream << std::endl;
	}

	template <typename... Args>
	static void log(
		std::string_view signature,
		Args&&... args
	) {
		log(signature, std::cerr, std::forward<Args>(args)...);
	}

private:
	static constexpr std::string_view extract_class(std::string_view signature) {
		size_t start = signature.find('(');
		if(start == std::string_view::npos) return "Unknown";

		std::string_view name = signature.substr(0, start);
		size_t method_pos = name.rfind("::");
		if(method_pos == std::string_view::npos) return "Main";

		size_t class_pos = name.rfind(' ', method_pos);
		if(class_pos == std::string_view::npos) class_pos = 0;
		else ++class_pos;

		return name.substr(class_pos, method_pos - class_pos);
	}
};

#endif // LOGGER_HPP