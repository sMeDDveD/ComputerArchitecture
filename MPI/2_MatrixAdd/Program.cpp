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

	static const int L;
	static const int M;

	static void Main() {
		auto process = MPI::Process();

		float* firstMatrix = nullptr;
		float* secondMatrix = nullptr;

		int* sliceLength = nullptr;
		int* sliceStart = nullptr;



		if (process.IsMaster()) {

			firstMatrix = GenerateMatrix();
			secondMatrix = GenerateMatrix();
			
			std::cout << "First matrix: " << std::endl;
			PrintMatrix(firstMatrix);
			std::cout << "Second matrix: " << std::endl;
			PrintMatrix(secondMatrix);


			int numOfProcess = process.ProcessCount();

			sliceLength = new int[numOfProcess];
			sliceStart = new int[numOfProcess];

			int length = L * M;

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

		MPI_Scatterv(firstMatrix, sliceLength, sliceStart,
			MPI_FLOAT, firstSlice, processSliceLength,
			MPI_FLOAT, MPI::MasterRank, MPI_COMM_WORLD);
		MPI_Scatterv(secondMatrix, sliceLength, sliceStart,
			MPI_FLOAT, secondSlice, processSliceLength,
			MPI_FLOAT, MPI::MasterRank, MPI_COMM_WORLD);

		MatrixAdd(firstSlice, secondSlice, processSliceLength);

		MPI_Gatherv(firstSlice, processSliceLength, MPI_FLOAT,
			firstMatrix, sliceLength, sliceStart,
			MPI_FLOAT, MPI::MasterRank, MPI_COMM_WORLD);


		if (process.IsMaster()) {
			std::cout << "Result: " << std::endl;
			PrintMatrix(firstMatrix);

			delete[] firstMatrix;
			delete[] secondMatrix;
			delete[] sliceStart;
			delete[] sliceLength;
		}
		delete[] firstSlice;
		delete[] secondSlice;
	}
private:
	static float* GenerateMatrix() {
		float* result = new float[L * M];
		for (int i = 0; i < L * M; i++) {
			result[i] = distribution(gen);
		}
		return result;
	}

	static void PrintMatrix(float* matrix) {
		for (int i = 0; i < L; i++) {
			for (int j = 0; j < M; j++) {
				std::cout << matrix[i * M + j] << '\t';
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}

	static void MatrixAdd(float* A, float* B, int size) {
		for (int i = 0; i < size; i++) {
			A[i] += B[i];
		}
	}
};

std::uniform_int_distribution<> Program::distribution(-1, 1);
std::random_device Program::rd;
std::mt19937 Program::gen(rd());

const int Program::L = 4;
const int Program::M = 4;

int main() {
	Program::Main();
	return 0;
}