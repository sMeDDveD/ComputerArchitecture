#include <mpi.h>
#include <iostream>

class Program
{
	static int _processCount;
	static int _rank;

public:
	static void Main(int argc, char* argv[])
	{
		MPI_Init(&argc, &argv);                        // 1: ������������� ��������� MPI � �������� ���������� ��������� ������.
													   // ��� ������� ������ ���� ������� �� ���� ������ ������� MPI.
													   // ��������� ���������� null, ���� ��������� �� �����.

		MPI_Comm_size(MPI_COMM_WORLD, &_processCount); // 2: ��������� ����� ���������.
		MPI_Comm_rank(MPI_COMM_WORLD, &_rank);		   // 3: ��������� ������� �������� (������ �� 0), � MPI �� ���������� �����.
													   // MPI_COMM_WORLD � ��� ������ ���� ���������. ��������� ������ � �������� ����������� �����.

		Hi();										   // 4: ����� � �������.
		Bye();										   // 5: ����� � �������. �������� �������� �� ������� ������.

		MPI_Finalize();								   // 6: ������������ �������� MPI. ��� ������� ������ ���������� ����� ���� ������ ������� MPI.
	}

private:
	static void Hi()
	{
		std::cout << "Hi from process #" << _rank << " (total: " << _processCount << ")!" << std::endl;
	}

	static void Bye()
	{
		std::cout << "Bye from process #" << _rank << "!" << std::endl;
	}
};

int Program::_processCount = 0;
int Program::_rank = 0;

int main(int argc, char* argv[])
{
	Program::Main(argc, argv);
	system("pause");
	return 0;
}