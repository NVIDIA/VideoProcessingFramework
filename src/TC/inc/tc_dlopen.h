#pragma once

#ifdef _WIN32
#include <windows.h>
#define TC_LIB HMODULE
#else
#define TC_LIB void*
#endif

TC_LIB tc_dlopen(const char* path);
void* tc_dlsym(TC_LIB lib, const char* symbol);
char* tc_dlerror(void);
int tc_dlclose(TC_LIB lib);
