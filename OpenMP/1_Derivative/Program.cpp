#include <iostream>
#include <algorithm>
#include <functional>
#include <iomanip>
#include <omp.h>
#include <sstream>
#include <cmath>

// Optimal N = 17000, h = 10e-5

namespace MyFunctions {
	constexpr double PI = 3.14159265358979;
	auto f = [](double x) {return sin(x) - x; };
	auto g = [](double x) {return 2 * x - 2; };
	auto h = [](double x) {return sin(x - 1); };

	auto iG = [](double a, double b) {return b * b - a * a - 2 * (b - a); };
	auto iH = [](double a, double b) {return cos(a - 1) - cos(b - 1); };
}

class Program {
public:
	using Function = std::function<double(double)>;
	static void Main(int numberOfThreads, int N) {
		PreSet(numberOfThreads);

		std::cout << "Result " << GetMaxDerivative
		(-MyFunctions::PI / 2, MyFunctions::PI / 2, N, MyFunctions::h) << std::endl;
		std::cout << "Real " << 1 << std::endl;

	}
private:
	static double GetMaxDerivative(double a, double b, int n, Function f) {
		double result = std::numeric_limits<double>::min();
		double prev, curr, next;
		prev = curr = next = std::nan("0");
		double h = (b - a) / n;

		#pragma omp parallel for firstprivate(prev, curr, next)			   // #pragma omp parallel for reduction(max: result), if OpenMP 3.0+ enabled
		for (int i = 0; i <= n; i++) {
			double cStep = h * i;										   // #pragma omp atomic if cStep += h
			if (std::isnan(prev)) {
				prev = f(cStep - h + a);
				curr = f(cStep + a);
			}
			next = f(cStep + h + a);
			double cResult = std::abs(prev - 2 * curr + next);
			if (cResult > result)
			{
				#pragma omp critical
				{
					if (cResult > result) {     
						result = cResult;
					}
				}
			}
			prev = curr;
			curr = next;
		}
		return result / (h * h);
	}

	static void PreSet(int n) {
		std::cout << std::setprecision(12);
		omp_set_num_threads(n);
		omp_set_nested(true);
	}
};

int main(int argc, char** argv) {
	if (argc != 3) {
		std::cout << "Wrong number of parametrs" << std::endl;
	}
	else {
		Program::Main(strtol(argv[1], nullptr, 10), strtol(argv[2], nullptr, 10));
	}
	system("pause");
	return 0;
}