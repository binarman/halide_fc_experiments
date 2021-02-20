#include "fc.h"

#include "HalideBuffer.h"

#include <chrono>
#include <iostream>

template <typename T>
void fill_buffer(Halide::Runtime::Buffer<T> b)
{
  size_t elems = b.number_of_elements();
  for (int i = 0; i < elems; ++i)
  {
    b.data()[i] = i;
  }
}

int main()
{
  const halide_filter_metadata_t *meta = fc_metadata();
  //(m,k) input1(a.k.a. weight) and (n,k) input0(a.k.a. input), which will produce (n,m) output
  int n = meta->arguments[0].buffer_estimates[1][1];
  int m = meta->arguments[2].buffer_estimates[0][1];
  int k = meta->arguments[0].buffer_estimates[0][1];

  Halide::Runtime::Buffer<float> input(k, n);
  Halide::Runtime::Buffer<float> bias(m);
  Halide::Runtime::Buffer<float> weights(k, m);
  fill_buffer(input);
  fill_buffer(bias);
  fill_buffer(weights);
  Halide::Runtime::Buffer<float> output(m, n);

  auto start = std::chrono::system_clock::now();

  int runs = 10000;
  for (int i = 0; i < runs; ++i)
    fc(input, weights, bias, output);

  auto finish = std::chrono::system_clock::now();

  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();

  std::cout << "elapsed: " << elapsed / runs << " microseconds";

  return 0;
}
