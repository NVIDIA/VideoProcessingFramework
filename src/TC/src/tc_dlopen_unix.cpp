#include <tc_dlopen.h>

#include <dlfcn.h>

TC_LIB tc_dlopen(const char* path) { return dlopen(path, RTLD_LAZY); }
void* tc_dlsym(TC_LIB lib, const char* symbol) { return dlsym(lib, symbol); }
char* tc_dlerror() { return dlerror(); }
int tc_dlclose(TC_LIB lib) { return dlclose(lib); }
