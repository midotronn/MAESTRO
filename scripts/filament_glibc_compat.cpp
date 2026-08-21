#include <cstdarg>
#include <cstdio>
#include <cstdlib>

extern "C" {

long __isoc23_strtol(const char* value, char** end, int base) {
    return std::strtol(value, end, base);
}

long long __isoc23_strtoll(const char* value, char** end, int base) {
    return std::strtoll(value, end, base);
}

unsigned long __isoc23_strtoul(const char* value, char** end, int base) {
    return std::strtoul(value, end, base);
}

unsigned long long __isoc23_strtoull(
        const char* value, char** end, int base) {
    return std::strtoull(value, end, base);
}

int __isoc23_sscanf(const char* input, const char* format, ...) {
    va_list arguments;
    va_start(arguments, format);
    const int result = std::vsscanf(input, format, arguments);
    va_end(arguments);
    return result;
}

}
