/////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2002 - 2022,
//  All rights reserved.
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions
//  are met:
//     1. Redistributions of source code must retain the above copyright
//        notice, this list of conditions and the following disclaimer.
//     2. Redistributions in binary form must reproduce the above copyright
//        notice, this list of conditions and the following disclaimer in the
//        documentation and/or other materials provided with the distribution.
//     3. The names of its contributors may not be used to endorse or promote
//        products derived from this software without specific prior written
//        permission.
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//	NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#define _CRT_SECURE_NO_WARNINGS

#include "SeLinearCG.h"
#include "SeSparseSystem.h"

#include "SeSchwarzPreconditioner.h"
#include "SeSchwarzPreconditionerPreviousVersion.h"
#include "eigen-3.4.1/Eigen/Eigen"
#include "eigen-3.4.1/unsupported/Eigen/SparseExtra"

#include <fstream>
#include <sstream>

#include <iostream>


SE_USING_NAMESPACE

bool LoadTriangleMesh(const std::string& filepath, std::vector<SeVec3f>& positions, std::vector<Int4>& faces)
{
	int vertNumber = 0;
	int faceNumber = 0;

	positions.clear();
	faces.clear();

	positions.reserve(1024);
	faces.reserve(2048);

	//====

	std::string line, prefix;

	std::ifstream file(filepath);

	if (!file.is_open())
		return false;

	while(std::getline(file, line))
	{
		std::istringstream iss(line);

		iss >> prefix;

		if (prefix == "v")
		{
			SeVec3f temp(0.f);

			iss >> temp.x >> temp.y >> temp.z;

			positions.emplace_back(temp);

			vertNumber++;
		}
		if (prefix == "f")
		{
			Int4 temp(0);

			std::string x;
			std::string y;
			std::string z;

			iss >> x >> y >> z;

			temp.x = std::stoi(x.substr(0, x.find_first_of("/") + 1));
			temp.y = std::stoi(y.substr(0, y.find_first_of("/") + 1));
			temp.z = std::stoi(z.substr(0, z.find_first_of("/") + 1));

			faces.emplace_back(temp - Int4(1));

			faceNumber++;
		}
	}

	file.close();

	printf("v: %d; t: %d\n", vertNumber, faceNumber);

	return true;
}

bool LoadTriangleMesh2(const std::string& filepath, std::vector<SeVec3f>& positions, std::vector<Int4>& faces)
{
	int vertNumber = 0;
	int faceNumber = 0;

	positions.clear();
	faces.clear();

	positions.reserve(1024);
	faces.reserve(2048);

	//====

	std::string line, prefix;

	std::ifstream file(filepath);

	if (!file.is_open())
		return false;

	while (std::getline(file, line))
	{
		std::istringstream iss(line);

		iss >> prefix;

		if (prefix == "v")
		{
			SeVec3f temp(0.f);

			iss >> temp.x >> temp.y >> temp.z;

			positions.emplace_back(temp);

			vertNumber++;
		}
		if (prefix == "f")
		{
			Int4 temp(0);

			std::string x;
			std::string y;
			std::string z;

			iss >> temp.x >> temp.y >> temp.z;

			faces.emplace_back(temp - Int4(1));

			faceNumber++;
		}
	}

	file.close();

	printf("v: %d; t: %d\n", vertNumber, faceNumber);

	return true;
}

bool LoadSparseMatrix(const std::string& filepath, std::vector<SeMat3f>& diagonals, std::vector<SeMat3f>& csrOffDiagonals, std::vector<int>& csrOffColIdx, std::vector<int>& csrRanges)
{
	printf("Loading Matrix...");

	diagonals.clear();
	csrOffDiagonals.clear();
	csrOffColIdx.clear();
	csrRanges.clear(); 

	std::string line, prefix;

	std::ifstream file(filepath);

	if (!file.is_open())
		return false;

	int M, N;
	int total = 0;
	int totalNNZ = 0;

	if (std::getline(file, line))
	{
		std::istringstream iss(line);

		iss >> M >> N >> totalNNZ;

		diagonals.reserve(std::max(M, N));
		csrRanges.reserve(std::max(M, N) + 1);
		csrOffDiagonals.reserve(totalNNZ);
		csrOffColIdx.reserve(totalNNZ);
	}

	csrRanges.push_back(total);

	while(std::getline(file, line))
	{
		std::istringstream iss(line);

		int num = 0; iss >> num;

		//==== diagonal

		int vId = 0; iss >> vId;

		SeMat3f diag;

		iss >> diag(0, 0) >> diag(0, 1) >> diag(0, 2)
			>> diag(1, 0) >> diag(1, 1) >> diag(1, 2)
			>> diag(2, 0) >> diag(2, 1) >> diag(2, 2);

		diagonals.push_back(diag);

		//==== off-diagonal

		for (int n = 1; n < num; ++n)
		{
			int colIdx = -1;

			iss >> colIdx; csrOffColIdx.push_back(colIdx);

			SeMat3f offDiag;

			iss >> offDiag(0, 0) >> offDiag(0, 1) >> offDiag(0, 2)
				>> offDiag(1, 0) >> offDiag(1, 1) >> offDiag(1, 2)
				>> offDiag(2, 0) >> offDiag(2, 1) >> offDiag(2, 2);

			csrOffDiagonals.push_back(offDiag);
		}

		total += num - 1; csrRanges.push_back(total);
	}

	printf("totalOffDiagonal: %d\n", total);

	file.close();

	return true;
}

bool LoadAndConvertSparseMatrix(const std::string& filepath, std::vector<SeMat3f>& diagonals, std::vector<SeMat3f>& csrOffDiagonals, std::vector<int>& csrOffColIdx, std::vector<int>& csrRanges)
{
	printf("Loading Matrix...\n");

	Eigen::SparseMatrix<double> lower_A_csc;

	if (!Eigen::loadMarket(lower_A_csc, filepath)) {
		std::cerr << "File " << filepath << " is not found" << std::endl;
	}


	Eigen::SparseMatrix<double> A_CSC =
		lower_A_csc.selfadjointView<Eigen::Lower>();

	int Number_of_nodes = A_CSC.rows() / 3;
	assert(A_CSC.rows() % 3 == 0);
	assert(A_CSC.nonZeros() % 9 == 0);
	int* Ap = A_CSC.outerIndexPtr();
	int* Ai = A_CSC.innerIndexPtr();
	std::vector<int> test(Ai, Ai + Ap[A_CSC.rows()]);

	double* Ax = A_CSC.valuePtr();

	std::vector<int> block_Ap;
	std::vector<int> block_Ai;
	std::vector<Eigen::Matrix3d> block_value;
	block_Ap.emplace_back(0);

	for (int i = 0; i < Number_of_nodes; i++) {

		Eigen::Matrix3d block;

		int x_row = i * 3;
		int y_row = i * 3 + 1;
		int z_row = i * 3 + 2;

		int x_NNZ = Ap[x_row + 1] - Ap[x_row];
		int y_NNZ = Ap[y_row + 1] - Ap[y_row];
		int z_NNZ = Ap[z_row + 1] - Ap[z_row];

		assert(x_NNZ == y_NNZ);
		assert(y_NNZ == z_NNZ);
		assert(x_NNZ % 3 == 0);

		// Adding the value
		for (int nz = 0; nz < x_NNZ / 3; nz++) {

			// Adding the values
			double* x_value = &Ax[Ap[x_row] + nz * 3];
			double* y_value = &Ax[Ap[y_row] + nz * 3];
			double* z_value = &Ax[Ap[z_row] + nz * 3];

			// Adding the col_idx
			block_Ai.emplace_back(Ai[Ap[x_row] + nz * 3] / 3);

			// First row
			block(0, 0) = x_value[0];
			block(0, 1) = x_value[1];
			block(0, 2) = x_value[2];
			// Second row
			block(1, 0) = y_value[0];
			block(1, 1) = y_value[1];
			block(1, 2) = y_value[2];
			// Third row
			block(2, 0) = z_value[0];
			block(2, 1) = z_value[1];
			block(2, 2) = z_value[2];

			block_value.emplace_back(block);
		}
		block_Ap.emplace_back(block_value.size());
	}

	diagonals.clear();
	csrOffDiagonals.clear();
	csrOffColIdx.clear();
	csrRanges.clear();

	int M, N;
	int total = 0;
	int totalNNZ = 0;

	M = Number_of_nodes;
	N = Number_of_nodes;
	totalNNZ = A_CSC.nonZeros() / 9 - Number_of_nodes;
		

	diagonals.reserve(std::max(M, N));
	csrRanges.reserve(std::max(M, N) + 1);
	csrOffDiagonals.reserve(totalNNZ);
	csrOffColIdx.reserve(totalNNZ);

	csrRanges.push_back(total);

	for (int vId = 0; vId < Number_of_nodes; vId++) {
		int num_of_nbr = block_Ap[vId + 1] - block_Ap[vId];
		int diag_idx = -1;
		for (int nbr_ptr = block_Ap[vId]; nbr_ptr < block_Ap[vId + 1];
			nbr_ptr++) {
			if (block_Ai[nbr_ptr] == vId) {
				diag_idx = nbr_ptr;
				break;
			}
		}
		assert(diag_idx != -1);

		//==== diagonal

		SeMat3f diag;

		Eigen::Matrix3d& block = block_value[diag_idx];

		// Write matrix elements to the file
		for (int row = 0; row < 3; row++) {
			for (int col = 0; col < 3; col++) {
				diag(row, col) = block(row, col);
			}
		}

		diagonals.push_back(diag);

		//==== off-diagonal

		for (int nbr_ptr = block_Ap[vId]; nbr_ptr < block_Ap[vId + 1];
			nbr_ptr++) {
			if (diag_idx == nbr_ptr) {
				continue;
			}
			int col_idx = block_Ai[nbr_ptr];
			csrOffColIdx.push_back(col_idx);
			block = block_value[nbr_ptr];
			SeMat3f offDiag;
			// Write matrix elements to the file
			for (int row = 0; row < 3; row++) {
				for (int col = 0; col < 3; col++) {
					offDiag(row, col) = block(row, col);
				}
			}
			csrOffDiagonals.push_back(offDiag);
		}

		total += num_of_nbr - 1;
		csrRanges.push_back(total);
	}

		printf("totalOffDiagonal: %d\n", total);

		return true;
}

bool LoadRhs(const std::string& filepath, std::vector<SeVec3f>& rhs)
{
	printf("Loading Rhs...\n");

	rhs.clear(); rhs.reserve(1024);

	std::string line, prefix;

	std::ifstream file(filepath);

	int vertNumber = 0;

	if (!file.is_open())
		return false;

	while (std::getline(file, line))
	{
		std::istringstream iss(line);

		iss >> prefix;

		if (prefix == "v")
		{
			SeVec3f temp(0.f);

			iss >> temp.x >> temp.y >> temp.z;

			rhs.push_back(temp);

			vertNumber++;
		}
	}

	printf("Total Vertex Number: %d\n", vertNumber);

	file.close();

	return true;
}

bool LoadRhs2(const std::string& filepath, std::vector<SeVec3f>& rhs)
{
	printf("Loading Rhs...\n");

	rhs.clear(); rhs.reserve(1024);

	std::string line, prefix;

	std::ifstream file(filepath);

	int vertNumber = 0;

	if (!file.is_open())
		return false;

	while (std::getline(file, line))
	{
		std::istringstream iss_x(line);

		SeVec3f temp(0.f);

		if (!(iss_x >> temp.x)) {
			std::cerr << "Error reading x-coordinate at line " << rhs.size() * 3 + 1 << std::endl;
			continue;
		}

		if (!(std::getline(file, line))) {
			std::cerr << "Error reading y-coordinate at line " << rhs.size() * 3 + 2 << std::endl;
			continue;
		}
		std::istringstream iss_y(line);
		if (!(iss_y >> temp.y)) {
			std::cerr << "Error reading y-coordinate at line " << rhs.size() * 3 + 2 << std::endl;
			continue;
		}

		if (!(std::getline(file, line))) {
			std::cerr << "Error reading z-coordinate at line " << rhs.size() * 3 + 3 << std::endl;
			continue;
		}
		std::istringstream iss_z(line);
		if (!(iss_z >> temp.z)) {
			std::cerr << "Error reading z-coordinate at line " << rhs.size() * 3 + 3 << std::endl;
			continue;
		}

		rhs.push_back(temp);

		vertNumber++;
	}

	printf("Total Vertex Number: %d\n", vertNumber);

	file.close();

	return true;
}


bool LoadEigenRHS(const std::string& filepath, Eigen::VectorXd& rhs)
{
	printf("Loading Eigen Rhs...\n");

	std::string line, prefix;

	std::ifstream file(filepath);

	int vertNumber = 0;

	if (!file.is_open())
		return false;

	int cnt = 0;
	while (std::getline(file, line))
	{
		std::istringstream iss_x(line);

		if (!(iss_x >> rhs[cnt])) {
			std::cerr << "Error reading x-coordinate at line " << cnt << std::endl;
			continue;
		}
		cnt++;
	}

	printf("Total Vertex Number: %d\n", cnt / 3);

	file.close();

	return true;
}

void BuildTopo(const std::vector<Int4>& faces, std::vector<Int4>& edges, int vertNumber)
{
	using IndexList = std::vector<int>;

	int faceNumber = SE_SCI(faces.size());

	edges.clear();
	edges.reserve(faceNumber * 2);

	std::vector<IndexList> vertAdjVerts(vertNumber);
	std::vector<IndexList> vertAdjEdges(vertNumber);

	for (auto& v : vertAdjVerts) { v.reserve(16); }
	for (auto& v : vertAdjEdges) { v.reserve(16); }

	auto pfnRecordEdge = [&](int vIdx0, int vIdx1) -> int
	{
		for (size_t i = 0; i < vertAdjVerts[vIdx0].size(); ++i)
		{
			if (vertAdjVerts[vIdx0][i] == vIdx1)
			{
				return vertAdjEdges[vIdx0][i];
			}
		}

		int eIdx = SE_SCI(edges.size());

		vertAdjVerts[vIdx0].emplace_back(vIdx1);
		vertAdjVerts[vIdx1].emplace_back(vIdx0);

		vertAdjEdges[vIdx0].emplace_back(eIdx);
		vertAdjEdges[vIdx1].emplace_back(eIdx);

		edges.emplace_back(vIdx0, vIdx1, -1, -1);

		return eIdx;
	};

	for (int fIdx = 0; fIdx < faceNumber; ++fIdx)
	{
		const Int4& face = faces[fIdx];

		int eIdx0 = pfnRecordEdge(face.y, face.z);
		int eIdx1 = pfnRecordEdge(face.z, face.x);
		int eIdx2 = pfnRecordEdge(face.x, face.y);

		if (edges[eIdx0][2] < 0) { edges[eIdx0][2] = face.x; }
		else					 { edges[eIdx0][3] = face.x; }
		if (edges[eIdx1][2] < 0) { edges[eIdx1][2] = face.y; }
		else					 { edges[eIdx1][3] = face.y; }
		if (edges[eIdx2][2] < 0) { edges[eIdx2][2] = face.z; }
		else					 { edges[eIdx2][3] = face.z; }
	}

	for (int vId = 0; vId < vertNumber; ++vId)
	{
		std::sort(vertAdjVerts[vId].begin(), vertAdjVerts[vId].end());
	}
}


int main(int argc, char** argv)
{
	std::string base = "C:\\Users\\Behrooz\\Desktop\\Schwarz_Compatible\\test\\";
	std::string obj_name = "test.obj";
	std::string hessian_name = "Hessian.mtx";
	std::string rhs_name = "RHS";

	//============================================================================================
	//==== prepare mesh information for collision handling
	std::vector<SeVec3f> positions;
	std::vector<Int4> edges;					// indices of the two adjacent vertices and the two opposite vertices of edges
	std::vector<Int4> faces;					// indices of the three adjacent vertices of faces, the fourth is useless


	if (!LoadTriangleMesh2(base + obj_name, positions, faces))
	{
		printf("mesh loading failed!");
		return 1;
	}

	BuildTopo(faces, edges, positions.size());

	int vertNumber = positions.size();
	int edgeNumber = edges.size();
	int faceNumber = faces.size();

	//=======================================================
	// =====================================
	//==== prepare the linear system

	std::shared_ptr<SeSparseSystem>	sparseSystem = std::make_shared<SeSparseSystem>();

	std::vector<SeMat3f> diagonals;
	std::vector<SeMat3f> csrOffDiagonals;
	std::vector<int>	 csrOffColIdx;
	std::vector<int>	 csrRanges;

	SeCsr<int> neighbours;	// the adjacent information of vertices stored in a csr format


	std::string hessian_address = base + hessian_name;
	if (!LoadAndConvertSparseMatrix(hessian_address, diagonals, csrOffDiagonals, csrOffColIdx, csrRanges))
	{
		printf("Matrix loading failed!");
		return 1;
	}


	std::vector<SeVec3f> rhs;

	if (!LoadRhs2(base + rhs_name, rhs))
	{
		printf("Rhs loading failed!");
		return 1;
	}

	sparseSystem->InitSystem(diagonals, csrOffDiagonals, csrOffColIdx, csrRanges);
	neighbours.InitIdxs(csrRanges, csrOffColIdx);

	//============================================================================================
	//==== init preconditioner

	std::shared_ptr<SeSchwarzPreconditioner> preconditioner = std::make_shared<SeSchwarzPreconditioner>();

	preconditioner->m_sparseSystem = sparseSystem;
	preconditioner->m_positions = positions.data();
	preconditioner->m_edges = edges.data();
	preconditioner->m_faces = faces.data();
	preconditioner->m_neighbours = &neighbours;


	preconditioner->Allocate(vertNumber, edgeNumber, faceNumber);
	preconditioner->Prepare();

	//============================================================================================
	//==== solve the linear system Ax=b by using PCG

	std::shared_ptr<SeLinearCG> linearCG = std::make_shared<SeLinearCG>(sparseSystem, preconditioner);

	constexpr int maxIter = 256;

	constexpr float tolerance = 1e-10f;

	constexpr bool useTolerance = false;

	std::vector<SeVec3f> x;   x.resize(vertNumber);	  Utility::MemsetZero(x);

	linearCG->Solve(x.data(), rhs.data(), vertNumber, maxIter, tolerance, useTolerance);


	//Jacobi Preconditioner
	printf("Loading Matrix...\n");

	Eigen::SparseMatrix<double> lower_A_csc;
	if (!Eigen::loadMarket(lower_A_csc, hessian_address)) {
		std::cerr << "File " << hessian_address << " is not found" << std::endl;
	}
	Eigen::VectorXd eigen_rhs(lower_A_csc.rows());
	LoadEigenRHS(base + rhs_name, eigen_rhs);
	Eigen::VectorXd eigen_x;

	Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::DiagonalPreconditioner<double> > solver;
	solver.setMaxIterations(256);
	solver.setTolerance(1e-10f);
	solver.compute(lower_A_csc);
	if (solver.info() != Eigen::Success) {
		std::cerr << "Decomposition failed" << std::endl;
		return 0;
	}
	eigen_x = solver.solve(eigen_rhs);

	// Solve Quality
	Eigen::VectorXd res =
		(eigen_rhs - lower_A_csc.selfadjointView<Eigen::Lower>() * eigen_x);
	double residual = res.norm();
	std::cout << "Jacobi Preconditioner Norm residual is: " << residual << std::endl;

	return 0;
}