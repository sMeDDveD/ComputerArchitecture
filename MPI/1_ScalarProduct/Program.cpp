#include <mpi.h>
#include <iostream>
#include <random>
#include "Process.h"

class Program
{
public:
	static std::uniform_int_distribution<int> distribution;
	static std::random_device rd;
	static std::mt19937 gen;

	static void Main() {
		auto process = MPI::Process();

		float* firstVector = nullptr;
		float* secondVector = nullptr;

		int* sliceLength = nullptr;
		int* sliceStart = nullptr;

		if (process.IsMaster()) {
			int length = 0;
			std::cout << "N: ";
			std::cin >> length;

			firstVector = GenerateVector(length);
			secondVector = GenerateVector(length);

			std::cout << "First vector: ";
			PrintVector(firstVector, length);
			std::cout << "Second vector: ";
			PrintVector(secondVector, length);


			int numOfProcess = process.ProcessCount();

			sliceLength = new int[numOfProcess];
			sliceStart = new int[numOfProcess];


			int sliceSize = length / numOfProcess;
			int lastSlice = length % numOfProcess;

			for (int i = 0; i < numOfProcess; i++) {
				sliceLength[i] = sliceSize;
				sliceStart[i] = sliceSize * i;
			}

			for (int i = 0; i < lastSlice; i++) {
				sliceLength[i]++;
				for (int j = i + 1; j < numOfProcess; j++) {
					sliceStart[j]++;
				}
			}
		}

		int processSliceLength = 0;
		MPI_Scatter(sliceLength, 1, MPI_INT, &processSliceLength, 1, MPI_INT, MPI::MasterRank, MPI_COMM_WORLD);

		float* firstSlice = new float[processSliceLength];
		float* secondSlice = new float[processSliceLength];

		MPI_Scatterv(firstVector, sliceLength, sliceStart,
			MPI_FLOAT, firstSlice, processSliceLength,
			MPI_FLOAT, MPI::MasterRank, MPI_COMM_WORLD);
		MPI_Scatterv(secondVector, sliceLength, sliceStart,
			MPI_FLOAT, secondSlice, processSliceLength,
			MPI_FLOAT, MPI::MasterRank, MPI_COMM_WORLD);

		float sliceProduct = ScalarProduct(firstSlice, secondSlice, processSliceLength);

		float resultProduct = 0;

		MPI_Reduce(&sliceProduct, &resultProduct, 1, MPI_FLOAT, MPI_SUM, MPI::MasterRank, MPI_COMM_WORLD);



		if (process.IsMaster()) {
			std::cout << "Result: " << resultProduct;

			delete[] firstVector;
			delete[] secondVector;
			delete[] sliceStart;
			delete[] sliceLength;
		}
		delete[] firstSlice;
		delete[] secondSlice;
	}
private:
	static float* GenerateVector(int length) {
		float* result = new float[length];
		for (int i = 0; i < length; i++) {
			result[i] = distribution(gen);
		}
		return result;
	}

	static void PrintVector(float* vector, int length) {
		for (int i = 0; i < length; i++) {
			std::cout << vector[i] << ' ';
		}
		std::cout << std::endl;
	}

	static float ScalarProduct(float* first, float* second, int length) {
		float result = 0;
		for (int i = 0; i < length; i++) {
			result += first[i] * second[i];
		}
		return result;
	}
};

std::uniform_int_distribution<> Program::distribution(-1, 1);
std::random_device Program::rd;
std::mt19937 Program::gen(rd());

int main() {
	Program::Main();
	return 0;
}