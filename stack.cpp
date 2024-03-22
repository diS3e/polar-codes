#include "stack.h"
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
