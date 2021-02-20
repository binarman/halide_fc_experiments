#include <iostream>
#include <filesystem>

#include <Halide.h>

void print_buffer(Halide::Buffer<float> buf)
{
  std::cout << "printing tensor\n";
  int h = buf.height();
  int w = buf.width();
  for (int i = 0; i < w; ++i)
  {
    for (int j = 0; j < h; ++j)
      std::cout << buf(i, j) << " ";
    std::cout << "\n";
  }
}

// description of sizes of Matmul: n, k, m
int sizes[][3] = {
{1, 2, 1536},
{1, 128, 6144},
{1, 1536, 128},
{1, 1536, 256},
{1, 1536, 64},
{1, 256, 6144},
{1, 40, 6144},
{1, 64, 6144},
{4, 1000, 64},
{4, 10137, 256},
{8, 1536, 64},
{8, 64, 1536},
{4, 256, 621},
{4, 3157, 384},
{4, 3157, 96},
{4, 384, 1000},
{4, 384, 10137},
{4, 500, 384},
{4, 64, 1536},
{4, 96, 4000}
};

int main( ) {
//  Halide::Func test_func;
//  Halide::Var x, y;
//  Halide::RDom ry(0, 1);
//
//  constexpr int w = 2;
//  constexpr int h = 3;
//  Halide::Buffer<float> input(w, h);
//  for (int i = 0; i < w; ++i)
//    for (int j = 0; j < h; ++j)
//      input.data()[i * h + j] = i * h + j;
//  print_buffer(input);
//
//  test_func(x, y) = input(y, x/2);
//  Halide::Buffer<float> output(h, w);
//  test_func.realize(output);
//  test_func.compile_to_lowered_stmt("/proc/self/fd/1", {input});
//
//  print_buffer(output);

// (m,k) input1(a.k.a. weight) and (n,k) input0(a.k.a. input), which will produce (n,m) output
  Halide::Type input_type(Halide::Type::Float, 32, 1);
  Halide::Type output_type(Halide::Type::Float, 32, 1);
  Halide::load_plugin("autoschedule_adams2019");
  Halide::load_plugin("autoschedule_li2018");
  Halide::load_plugin("autoschedule_mullapudi2016");

  for (int *size: sizes)
  {
    int n = size[0];
    int k = size[1];
    int m = size[2];
    std::string name = "n" + std::to_string(n) + "m" + std::to_string(m) + "k" + std::to_string(k);
    std::cout << "processing " << name << "\n";
    Halide::Var i, j;
    Halide::RDom k_dom(0, k);
    Halide::ImageParam input(input_type, 2);
    Halide::ImageParam weights(input_type, 2);
    Halide::ImageParam bias(output_type, 1);
    Halide::Func fc;

    fc(j, i) = bias(j);
    fc(j, i) += input(k, i) * weights(k, j);

    input.set_estimates({{0, k}, {0, n}});
    bias.set_estimates({{0, m}});
    weights.set_estimates({{0, k}, {0, m}});

    fc.set_estimates({{0, m}, {0, n}});

    Halide::Pipeline pipeline(fc);

    std::string scheduler_name = "Adams2019";
//    std::string scheduler_name = "Li2018";
//    std::string scheduler_name = "Mullapudi2016";
    Halide::Target target;
    target.arch = Halide::Target::ARM;
    target.bits = 32;
    target.os = Halide::Target::Linux;
    target.set_features({Halide::Target::NoAsserts});
    Halide::MachineParams machine_params{1, 512*1024, 0.4f};
    auto scheduler_results = pipeline.auto_schedule(scheduler_name, target, machine_params);

    Halide::Module module = pipeline.compile_to_module({input, weights, bias}, "fc", target);
    module.set_auto_scheduler_results(scheduler_results);

    std::filesystem::create_directory(name);
    std::map<Halide::Output, std::string> output_files;
    output_files[Halide::Output::object] = std::filesystem::path(name) / "fc.o";
    output_files[Halide::Output::c_header] = std::filesystem::path(name) / "fc.h";
    module.compile(output_files);
  }
  return 0;
}
