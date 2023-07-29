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

#pragma once

#include "SeMatrix.h"

#include <string>

SE_NAMESPACE_BEGIN

class SeSparseSystem
{
public:

	const SeMat3f& Diagonal(int i) const { return m_diagonals[i]; }

	const SeMat3f& Entry(int i, int k) const { return m_csrOffDiagonals[m_csrRanges[i] + k]; }

public:

	const SeMat3f* Diagonals() const { return m_diagonals.data(); }

	const SeMat3f* CsrOffDiagonals() const { return m_csrOffDiagonals.data(); }

	const int*	   CsrRanges() const { return m_csrRanges.data(); }

public:

	void InitSystem(const std::vector<SeMat3f>& diagonals, const std::vector<SeMat3f>& csrOffDiagonals,
					const std::vector<int>& csrOffColIdx, const std::vector<int>& csrRanges)
	{
		m_diagonals = diagonals;
		m_csrOffDiagonals = csrOffDiagonals;
		m_csrOffColIdx = csrOffColIdx;
		m_csrRanges = csrRanges;
	}

public:

	virtual void Multiply(SeVec3f* ans, const SeVec3f* var, int dim) 
	{
		OMP_PARALLEL_FOR

			for (int i = 0; i < dim; ++i)
			{
				SeVec3f result = m_diagonals[i] * var[i];

				int begin = m_csrRanges[i];
				int end = m_csrRanges[i + 1];

				for (int j = begin; j < end; ++j)
				{
					int colIdx = m_csrOffColIdx[j];

					result += m_csrOffDiagonals[j] * var[colIdx];
				}

				ans[i] = result;
			}
	}

	virtual void Residual(SeVec3f* residual, const SeVec3f* var, const SeVec3f* rhs, int dim)
	{
		// residual = A * var;
		Multiply(residual, var, dim);

		// residual = rhs - A * var;
		OMP_PARALLEL_FOR

			for (int i = 0; i < dim; ++i)
			{
				residual[i] = rhs[i] - residual[i];
			}
	}

private:

	std::vector<SeMat3f> m_diagonals;

	std::vector<SeMat3f> m_csrOffDiagonals;

	std::vector<int>	 m_csrOffColIdx;

	std::vector<int>	 m_csrRanges;
};

SE_NAMESPACE_END