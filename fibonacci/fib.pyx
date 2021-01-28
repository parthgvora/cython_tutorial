
# Returns the nth fibonacci number
# Pretty much everything can be typed

cpdef int fibonacci(int n):
    cdef int f0
    cdef int f1
    cdef int f2
    cdef int i

    if n < 1:
        return 0

    f0 = 0
    f1 = 1
    for i in range(1, n):
        f2 = f0 + f1
        f0 = f1
        f1 = f2

    return f2


