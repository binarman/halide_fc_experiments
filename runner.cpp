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

void print_estimates(std::string name, int n, int64_t const *const *est)
{
  std::cout << "estimates of " << name << " ";
  for (int i = 0; i < n; ++i)
    std::cout << "{" << *est[i*2] << ", " << *est[i*2 + 1] << "} ";
  std::cout << "\n";
}

int main()
{
  const halide_filter_metadata_t *meta = fc_metadata();
  //(m,k) input1(a.k.a. weight) and (n,k) input0(a.k.a. input), which will produce (n,m) output
  int n = *meta->arguments[0].buffer_estimates[3];
  int m = *meta->arguments[2].buffer_estimates[1];
  int k = *meta->arguments[0].buffer_estimates[1];

  std::cout << "n" << n << "k" << k << "m" << m << " ";
//  std::cout << "n: " << n << "\nm: " << m << "\nk: " << k << "\n";
//  print_estimates("input", meta->arguments[0].dimensions, meta->arguments[0].buffer_estimates);
//  print_estimates("weights", meta->arguments[1].dimensions, meta->arguments[1].buffer_estimates);
//  print_estimates("bias", meta->arguments[2].dimensions, meta->arguments[2].buffer_estimates);

  Halide::Runtime::Buffer<int8_t> input(k, n);
  Halide::Runtime::Buffer<int32_t> bias(m);
  Halide::Runtime::Buffer<int8_t> weights(k, m);
  fill_buffer(input);
  fill_buffer(bias);
  fill_buffer(weights);
  Halide::Runtime::Buffer<int32_t> output(m, n);

  auto start = std::chrono::system_clock::now();

  constexpr int runs = 1000;
  for (int i = 0; i < runs; ++i)
    fc(input.raw_buffer(), weights.raw_buffer(), bias.raw_buffer(), output.raw_buffer());

  auto finish = std::chrono::system_clock::now();

//  for (int i = 0; i < 4; i++)
//    std::cout << output.data()[i] << "\n";

  float elapsed = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();

  std::cout << elapsed / runs << "\n";

  return 0;
}
