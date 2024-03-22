#ifndef POLAR_CODES_STACK_H
#define POLAR_CODES_STACK_H
#include <stdexcept>

template <typename T>
class stack {
private:
    T* memory;
    int _size = 0;
    int _capacity;
public:
    stack(int max_size);

    stack(const stack<T>& other_stack);

    ~stack();

    void push(T value);

    T pop();

    int size();

    void clear();

    int capacity();
};


#endif //POLAR_CODES_STACK_H
