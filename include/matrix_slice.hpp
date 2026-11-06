#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>
#include <type_traits>

enum class SliceAxis {
    Row,
    Column
};

class NonContiguousSliceError : public std::logic_error {
public:
    explicit NonContiguousSliceError(const char* message)
        : std::logic_error(message) {}
};

template <class T>
class MatrixSlice {
    static_assert(std::is_same_v<std::remove_const_t<T>, double>);

public:
    MatrixSlice() noexcept = default;

    MatrixSlice(
        T* data,
        std::size_t size,
        std::size_t stride,
        SliceAxis axis
    )
        : data_(size == 0 ? nullptr : data),
          size_(size),
          stride_(stride),
          axis_(axis)
    {
        if (size_ > 0 && data_ == nullptr) {
            throw std::invalid_argument(
                "Non-empty matrix slice requires a valid pointer"
            );
        }

        if (size_ > 0 && stride_ == 0) {
            throw std::invalid_argument(
                "Matrix slice stride must be positive"
            );
        }
    }

    std::size_t size() const noexcept {
        return size_;
    }

    bool empty() const noexcept {
        return size_ == 0;
    }

    std::size_t stride() const noexcept {
        return stride_;
    }

    SliceAxis axis() const noexcept {
        return axis_;
    }

    bool is_contiguous() const noexcept {
        return size_ <= 1 || stride_ == 1;
    }

    T* data() const noexcept {
        return data_;
    }

    T& operator()(std::size_t index) const noexcept {
        return data_[index * stride_];
    }

    T& at(std::size_t index) const {
        if (index >= size_) {
            throw std::out_of_range(
                "Matrix slice index out of range"
            );
        }

        return (*this)(index);
    }

    std::span<T> operator*() const {
        if (!is_contiguous()) {
            throw NonContiguousSliceError(
                "Cannot convert non-contiguous slice to span"
            );
        }

        return {data_, size_};
    }

    std::span<T> as_span() const {
        return operator*();
    }

    MatrixSlice<const double> as_const() const noexcept {
        return MatrixSlice<const double>(
            data_, size_, stride_, axis_
        );
    }

private:
    T* data_ = nullptr;
    std::size_t size_ = 0;
    std::size_t stride_ = 1;
    SliceAxis axis_ = SliceAxis::Row;
};
