#include <tc_dlopen.h>
#include <windows.h>

TC_LIB tc_dlopen(const char* path) { return LoadLibraryA (path); }
void* tc_dlsym(TC_LIB lib, const char* symbol) { return GetProcAddress(lib, symbol); }
char* tc_dlerror() { return (char*)"Failed"; }
int tc_dlclose(TC_LIB lib) { return FreeLibrary(lib); }
