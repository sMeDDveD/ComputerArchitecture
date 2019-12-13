#include "Matrix.h"

#include <exception>
#include <random>

Matrix::Matrix(int rows, int columns, double* data) : rows(rows), columns(columns), Data(data) { }

Matrix::Matrix(int rows, int columns) : rows(rows), columns(columns), Data(new double[rows * columns]{}) { }

void Matrix::Clear()
{
	delete[] Data;
	Data = nullptr;
}

double Matrix::Get(int i, int j) const
{
	return Data[i * columns + j];
}

int Matrix::GetRows() const
{
	return rows;
}

int Matrix::GetColumns() const
{
	return columns;
}

Matrix Matrix::Transpose() const
{
	auto data = new double[rows * columns];
	for (auto i = 0; i < rows; ++i)
	{
		for (auto j = 0; j < columns; ++j)
		{
			data[j * rows + i] = Get(i, j);
		}
	}
	return Matrix(columns, rows, data);
}

Matrix Matrix::operator* (const Matrix& matrix) const
{
	if (this->GetColumns() != matrix.GetRows())
		throw std::exception("Matrixes are not agreed");

	auto n = this->GetRows();
	auto m = this->GetColumns();
	auto k = matrix.GetColumns();

	auto data = new double[n * k];

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < k; j++)
		{
			double sum = 0;
			for (int q = 0; q < m; q++)
				sum += this->Get(i, q) * matrix.Get(q, j);

			data[i * k + j] = sum;
		}
	}

	return Matrix(n, k, data);
}

std::ofstream & operator<<(std::ofstream & outputStream, const Matrix & matrix)
{
	for (auto i = 0; i < matrix.GetRows(); ++i)
	{
		for (auto j = 0; j < matrix.GetColumns(); ++j)
		{
			outputStream << std::fixed << static_cast<int>(matrix.Get(i, j)) << " ";
		}

		outputStream << std::endl;
	}

	outputStream << std::endl;

	return outputStream;
}

Matrix Matrix::GenerateMatrix(int n, int m)
{
	std::uniform_int_distribution<> distribution(-1, 1);
	std::random_device rd;
	std::mt19937 gen(rd());

	auto size = n * m;
	auto matrix = new double[size];

	for (auto k = 0; k < size; ++k)
	{
		matrix[k] = distribution(gen);
	}

	return Matrix(n, m, matrix);
}
