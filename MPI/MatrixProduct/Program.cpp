#include <mpi.h>
#include <iostream>
#include <random>
#include "Process.h"

namespace MPI
{
	class Timer
	{
		double _startTime;

	public:
		Timer() { _startTime = MPI_Wtime(); }
		double Seconds() const { return MPI_Wtime() - _startTime; }
	};
}

class Program
{
public:
	static std::uniform_int_distribution<int> distribution;
	static std::random_device rd;
	static std::mt19937 gen;

	static void Main(int K, int L, int M) {
		auto process = MPI::Process();
		auto time = MPI::Timer();

		float* firstMatrix = nullptr;
		float* secondMatrix = new float[L * M];

		int* sliceLengthBefore = nullptr;
		int* sliceStartBefore = nullptr;

		int* sliceLengthAfter = nullptr;
		int* sliceStartAfter = nullptr;

		float* answerMatrix = new float[K * M];

		if (process.IsMaster()) {
			firstMatrix = GenerateMatrix(K, L);
			secondMatrix = GenerateMatrix(L, M);
				
			std::cout << "First matrix: " << std::endl;
			PrintMatrix(firstMatrix, K, L);
			std::cout << "Second matrix: " << std::endl;
			PrintMatrixT(secondMatrix, L, M);

			int numOfProcess = process.ProcessCount();

			sliceLengthBefore = new int[numOfProcess];
			sliceStartBefore = new int[numOfProcess];

			int sliceSize = K / numOfProcess;
			int lastSlice = K % numOfProcess;

			for (int i = 0; i < numOfProcess; i++) {
				sliceLengthBefore[i] = sliceSize * L;
				sliceStartBefore[i] = sliceSize * i * L;
			}

			for (int i = 0; i < lastSlice; i++) {
				sliceLengthBefore[i] += L;
				for (int j = i + 1; j < numOfProcess; j++) {
					sliceStartBefore[j] += L;
				}
			}
			
			sliceLengthAfter = new int[numOfProcess];
			sliceStartAfter = new int[numOfProcess];

			for (int i = 0; i < numOfProcess; i++) {
				sliceLengthAfter[i] = (sliceLengthBefore[i] / L) * M;
				sliceStartAfter[i] = (sliceStartBefore[i] / L) * M;
			}
		}
		MPI_Bcast(secondMatrix, L * M, MPI_FLOAT, MPI::MasterRank, MPI_COMM_WORLD);

		int processSliceLengthBefore = 0;
		MPI_Scatter(sliceLengthBefore, 1, MPI_INT, &processSliceLengthBefore, 1, MPI_INT, MPI::MasterRank, MPI_COMM_WORLD);

		float* sliceLines = new float[processSliceLengthBefore];

		MPI_Scatterv(firstMatrix, sliceLengthBefore, sliceStartBefore,
			MPI_FLOAT, sliceLines, processSliceLengthBefore,
			MPI_FLOAT, MPI::MasterRank, MPI_COMM_WORLD);


		MatrixProduct(sliceLines, secondMatrix, processSliceLengthBefore / L, L, M);

		int processSliceLengthAfter = (processSliceLengthBefore / L) * M;
		MPI_Gatherv(sliceLines, processSliceLengthAfter, MPI_FLOAT,
			answerMatrix, sliceLengthAfter, sliceStartAfter,
			MPI_FLOAT, MPI::MasterRank, MPI_COMM_WORLD);


		if (process.IsMaster()) {
			std::cout << "Result: " << std::endl;
			std::cout << "Time: " << time.Seconds() << std::endl;
			PrintMatrix(answerMatrix, K, M);

			delete[] firstMatrix;
			delete[] sliceStartBefore;
			delete[] sliceLengthBefore;
			delete[] sliceLengthAfter;
			delete[] sliceStartAfter;
		}
		delete[] secondMatrix;
		delete[] sliceLines;
	}
private:
	static float* GenerateMatrix(int L, int M) {
		float* result = new float[L * M];
		for (int i = 0; i < L * M; i++) {
			result[i] = distribution(gen);
		}
		return result;
	}

	static void PrintMatrix(float* matrix, int L, int M) {
		for (int i = 0; i < L; i++) {
			for (int j = 0; j < M; j++) {
				std::cout << matrix[i * M + j] << '\t';
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
	
	static void PrintMatrixT(float* matrix, int L, int M) {
		for (int j = 0; j < L; j++) {
			for (int i = 0; i < M; i++) {
				std::cout << matrix[i * L + j] << '\t';
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}

	static void MatrixProduct(float*& lines, float* B, int K, int L, int M) {
		float *answer = new float[K * M]();
		for (int i = 0; i < K; i++) {
			for (int j = 0; j < M; j++) {
				for (int w = 0; w < L; w++) {
					answer[i * M + j] += lines[i * L + w] * B[j * L + w];
				}
			}
		}
		delete[] lines;
		lines = answer;
	}
};

std::uniform_int_distribution<> Program::distribution(-1, 1);
std::random_device Program::rd;
std::mt19937 Program::gen(rd());

int main(int argc, char* argv[]) {
	Program::Main(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
	system("pause");
	return 0;
}