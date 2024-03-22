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

template<typename T>
stack<T>::stack(int max_size):
        _capacity(max_size) {
    memory = (T*) malloc(sizeof(T) * max_size);
}

template<typename T>
stack<T>::stack(const stack<T>& other_stack) {
    memory = (T*) malloc(sizeof(T) * other_stack._capacity);
    for (int i = 0; i < other_stack._size; ++i) {
        memory[other_stack._size - 1 - i] = other_stack.memory[i];
    }
}

template<typename T>
stack<T>::~stack() {
    free(memory);
}
template<typename T>
void stack<T>::push(T value) {
    memory[_size++] = value;
}
template<typename T>
T stack<T>::pop() {
    if (_size <= 0) {
        throw std::runtime_error("Pop from empty stack");
    }
    return memory[--_size];
}

template<typename T>
int stack<T>::size() {
    return _size;
}

template<typename T>
void stack<T>::clear() {
    _size = 0;
}

template<typename T>
int stack<T>::capacity() {
    return _capacity;
}


#endif //POLAR_CODES_STACK_H
