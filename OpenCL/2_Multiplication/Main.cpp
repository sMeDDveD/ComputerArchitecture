#include "OpenCL.h"
#include "DeviceEx.h"
#include "Matrix.h"
#include "KernelEx.h"

#include <iomanip>
#include <iostream>
#include <fstream>
#include <random>

using namespace cl;

const std::string defaultCl = "Kernel.cl";
const std::string deviceName = "Intel";

const std::string outputFile = "output.txt";
const std::string loggingFile = "time.txt";

class Main
{
public:

	static void Run(int argc, char** argv)
	{
		if (argc == 5)
		{
			Setting(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), argv[4]);
		} 
		else 
		{
			Setting(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
		}
	}

private:

	static void Setting(int N, int M, int L, const std::string& kernelName = "SM")
	{
		try
		{
			Device device = DeviceEx::Find(deviceName);
			RunOnDevice(device, N, M, L, kernelName);
		}
		catch (const std::exception &ex)
		{
			std::cerr << ex.what() << std::endl;
		}
	}

	static void RunOnDevice(const Device& device, int N, int M, int L, const std::string& kernelName)
	{
		Matrix firstMatrix = Matrix::GenerateMatrix(N, M);
		Matrix secondMatrix = Matrix::GenerateMatrix(L, M);

		std::ofstream fout(outputFile);

		printVectorSize(fout, device);
		 
		fout << "First matrix:" << std::endl;
		fout << firstMatrix;
		fout << "Second matrix:" << std::endl;
		fout << secondMatrix;

		secondMatrix = secondMatrix.Transpose();
		Context context = DeviceEx::CreateContext(device);
		Buffer firstBuffer = CreateBuffer(context, firstMatrix, CL_MEM_READ_ONLY);
		Buffer secondBuffer = CreateBuffer(context, secondMatrix, CL_MEM_READ_ONLY);

		Matrix answerMatrix = RunKernel(device, context, firstBuffer, secondBuffer, N, M, L, kernelName);

		fout << "Answer matrix" << std::endl;
		fout << answerMatrix;

		firstMatrix.Clear();
		secondMatrix.Clear();
		answerMatrix.Clear();
		fout.close();
	}

	static Matrix RunKernel(const Device& device, const Context& context, Buffer firstBuffer, 
		Buffer secondBuffer, int N, int M, int L, const std::string& kernelName)
	{
		Matrix answerMatrix = Matrix(N, L);
		Buffer answerBuffer = CreateBuffer(context, answerMatrix, CL_MEM_READ_WRITE);

		Kernel kernel = KernelEx::BuildFromFile(device, context, defaultCl, kernelName);

		kernel.setArg(0, firstBuffer);
		kernel.setArg(1, secondBuffer);
		kernel.setArg(2, answerBuffer);
		kernel.setArg(3, N);
		kernel.setArg(4, M);
		kernel.setArg(5, L);

		Event event = Event();
		CommandQueue queue = CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
		queue.enqueueNDRangeKernel
			(kernel, NullRange, NDRange(answerMatrix.GetRows(), answerMatrix.GetColumns()), NullRange, nullptr, &event);
		event.wait();
		queue.finish();

		cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();

		std::ofstream fout = std::ofstream(loggingFile, std::ios::app);
		fout << "T(" << kernelName << "-" << N << "-" << M << "-" << L << "): " 
			<< std::setprecision(5) << (end - start) / 1000000.0 << " millis" << std::endl;

		ReadFromBuffer(queue, answerMatrix, answerBuffer);
		fout.close();
		
		return answerMatrix;
	}

	static Buffer CreateBuffer(const Context& context, Matrix matrix, int accessFlag)
	{
		return Buffer(context, accessFlag | CL_MEM_COPY_HOST_PTR, matrix.GetRows() * matrix.GetColumns() 
			* sizeof(double), matrix.Data);
	}

	static void ReadFromBuffer(const CommandQueue& queue, Matrix matrix, const Buffer& buffer)
	{
		queue.enqueueReadBuffer(buffer, CL_TRUE, 0, matrix.GetRows() * matrix.GetColumns() 
			* sizeof(double), matrix.Data);
	}

	static void printVectorSize(std::ofstream& os, const Device& device) {
		os << "Preferred vector width: ";
		os << device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR>() << std::endl;
	}
};

int main(int argc, char** argv)
{
	Main::Run(argc, argv);
	system("pause");
}