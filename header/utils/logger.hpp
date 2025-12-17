#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#include <algorithm>

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
	static void log(std::string_view signature, Args&&... args) {
		log(signature, std::cerr, std::forward<Args>(args)...);
	}

	template <typename... Args>
	static void log(std::string_view signature, std::ostream& stream, Args&&... args) {
		std::string class_name = extract_class(signature);
		stream << "[" << class_name << "] ";
		((stream << std::forward<Args>(args) << ' '), ...);
		stream << std::endl;
	}

private:
	static std::string extract_class(std::string_view signature) {
		// 1. Check for GCC/Clang style "[with T = Type]" suffix
		// Example: "void Class<T>::method() [with T = int]"
		std::vector<std::pair<std::string, std::string>> replacements;
		size_t with_pos = signature.find("[with ");
		
		if (with_pos != std::string_view::npos) {
			size_t end_bracket = signature.find(']', with_pos);
			if (end_bracket != std::string_view::npos) {
				std::string_view params = signature.substr(with_pos + 6, end_bracket - (with_pos + 6));
				
				// Parse "Key = Value; Key2 = Value2"
				size_t current = 0;
				while (current < params.length()) {
					size_t eq_pos = params.find(" = ", current);
					if (eq_pos == std::string_view::npos) break;

					std::string key(params.substr(current, eq_pos - current));
					
					size_t semi_pos = params.find("; ", eq_pos);
					size_t val_end = (semi_pos == std::string_view::npos) ? params.length() : semi_pos;
					std::string val(params.substr(eq_pos + 3, val_end - (eq_pos + 3)));

					replacements.push_back({key, val});

					if (semi_pos == std::string_view::npos) break;
					current = semi_pos + 2;
				}
			}
		}

		// 2. Identify the main method signature (strip args and return type)
		// Cut off arguments based on the first '(' found before the [with ...] block
		size_t args_start = signature.find('(');
		if (args_start > with_pos && with_pos != std::string_view::npos) args_start = std::string_view::npos; // Handle weird edge cases

		std::string_view prefix = (args_start == std::string_view::npos) ? signature : signature.substr(0, args_start);

		// 3. Find the method separator "::"
		size_t method_pos = prefix.rfind("::");
		if (method_pos == std::string_view::npos) return "Main";

		// 4. Robust Backwards Search for Class Name
		// Scans backwards from "::" skipping spaces ONLY if they are inside brackets <...>
		// This handles "void Class<struct Type>::method" (MSVC style) correctly.
		size_t class_start = 0;
		int bracket_depth = 0;

		for (size_t i = method_pos; i > 0; --i) {
			char c = prefix[i - 1];
			
			if (c == '>') bracket_depth++;
			else if (c == '<') bracket_depth--;
			
			// If we hit a space at depth 0, we found the return type separator (e.g. "void Class")
			if (c == ' ' && bracket_depth == 0) {
				class_start = i;
				break;
			}
		}

		std::string class_name(prefix.substr(class_start, method_pos - class_start));

		// 5. Apply Template Replacements (GCC/Clang)
		for (const auto& kv : replacements) {
			size_t pos = 0;
			while ((pos = class_name.find(kv.first, pos)) != std::string::npos) {
				class_name.replace(pos, kv.first.length(), kv.second);
				pos += kv.second.length();
			}
		}

		// 6. Cleanup Prefixes (struct, class, etc.).
		const std::string prefixes[] = { "struct ", "class ", "void ", "static " };
		for (const auto& pre : prefixes) {
			size_t pos = 0;
			while ((pos = class_name.find(pre, pos)) != std::string::npos) {
				class_name.erase(pos, pre.length());
				// Don't advance pos, we might have created another match or need to re-check
			}
		}

		return class_name;
	}
};

#endif // LOGGER_HPP