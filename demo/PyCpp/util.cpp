#include "util.h"
#include <iostream>
#include <numpy/arrayobject.h>
namespace ipc {
        
    PyObject *
    PyByteArray_FromString_WithoutCopy(char *bytes, Py_ssize_t size)
    {
        PyByteArrayObject *arrayObject;
        Py_ssize_t alloc;

        if (size < 0) {
            PyErr_SetString(PyExc_SystemError,
                "Negative size passed to PyByteArray_FromStringAndSize");
            return NULL;
        }

        /* Prevent buffer overflow when setting alloc to size+1. */
        if (size == PY_SSIZE_T_MAX) {
            return PyErr_NoMemory();
        }

        arrayObject = PyObject_New(PyByteArrayObject, &PyByteArray_Type);
        if (arrayObject == NULL)
            return NULL;

        if (size == 0) {
            arrayObject->ob_bytes = NULL;
            alloc = 0;
        }
        else {
            arrayObject->ob_bytes = bytes;
        }
        Py_SIZE(arrayObject) = size;
        arrayObject->ob_alloc = alloc;
        arrayObject->ob_start = arrayObject->ob_bytes;
        arrayObject->ob_exports = 0;

        return (PyObject *)arrayObject;
    }

    PyObject *
    PyArray_FromIntArray(int *rind, Py_ssize_t size) {
        import_array();
        npy_intp dim[1];
        dim[0] = size;
        PyArrayObject *mat = (PyArrayObject*) PyArray_SimpleNewFromData(1, dim, NPY_INT, (char*) rind);
        return PyArray_Return(mat);
    }

}

