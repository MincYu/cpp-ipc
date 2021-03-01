#include<python3.6/Python.h>

namespace ipc {
	PyObject * PyByteArray_FromString_WithoutCopy(char *bytes, Py_ssize_t size);
	PyObject * PyArray_FromIntArray(int *rind, Py_ssize_t size);
}

