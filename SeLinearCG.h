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

#include "SeUtility.h"
#include "SeVectorSimd.h"
#include "SePreconditioner.h"

SE_NAMESPACE_BEGIN

class SeLinearCG
{
private:

	std::vector<SeVec3f> x;
	std::vector<SeVec3f> residual;
	std::vector<SeVec3f> p;
	std::vector<SeVec3f> Ap;
	std::vector<SeVec3f> z;

	SeVec3f* ptrX = nullptr;
	SeVec3f* ptrResidual = nullptr;
	SeVec3f* ptrP = nullptr;
	SeVec3f* ptrAp = nullptr;
	SeVec3f* ptrZ = nullptr;

	std::shared_ptr<SeSparseSystem>   m_sparseSystem;
	std::shared_ptr<SePreconditioner> m_preconditioner;

public:

	SeLinearCG(std::shared_ptr<SeSparseSystem> sparseSystem, std::shared_ptr<SePreconditioner> preconditioner)
	{
		m_sparseSystem = sparseSystem;
		m_preconditioner = preconditioner;
	}

	const SeVec3f* Solve(const SeVec3f* rhs, int dim, int maxIter, float tolerance, bool useTolerance)
	{
		Solve(ptrX, rhs, dim, maxIter, tolerance, useTolerance);

		return ptrX;
	}

	void Solve(SeVec3f* ans, const SeVec3f* rhs, int dim, int maxIter, float tolerance, bool useTolerance)
	{
		if (p.size() != dim)
		{
			Resize(dim);
		}

		Utility::MemsetZero(ans, dim);

		if (m_preconditioner)
			SolvePreconditioned(ans, rhs, dim, maxIter, tolerance, useTolerance);
		else
			SolveTrivial(ans, rhs, dim, maxIter, tolerance, useTolerance);
	}

private:

	void Resize(int dim)
	{
		Utility::ResizeAndShrink(x, dim);		 Utility::MemsetZero(x);
		Utility::ResizeAndShrink(residual, dim); Utility::MemsetZero(residual);
		Utility::ResizeAndShrink(p, dim);		 Utility::MemsetZero(p);
		Utility::ResizeAndShrink(Ap, dim);		 Utility::MemsetZero(Ap);
		Utility::ResizeAndShrink(z, dim);		 Utility::MemsetZero(z);

		ptrX = x.data();
		ptrResidual = residual.data();
		ptrP = p.data();
		ptrAp = Ap.data();
		ptrZ = z.data();
	}

	void SolveTrivial(SeVec3f* ans, const SeVec3f* rhs, int dim, int maxIter, float tolerance, bool useTolerance)
	{
		float rDot0 = 0.f;
		int it = 0;

		m_sparseSystem->Residual(ptrResidual, ans, rhs, dim);							// r_0 = b - A * x_0

		while (it++ < maxIter || useTolerance)
		{
			float rDot = Utility::VectorDot(ptrResidual, ptrResidual, dim);			// r_k^T * r_k

			float beta = rDot / rDot0;													// beta_k = rDot_k / rDot_k-1
			if (rDot0 == 0.f) { beta = 0.f; }

			//printf("It: %d;   rDot0: %f;   rDot1: %f\n", it, rDot0, rDot);

			rDot0 = rDot; if (rDot < tolerance) { break; }

			Utility::VectorSelfAdd(ptrP, ptrResidual, beta, 1.f, dim);				// p_k = beta_k * p_k-1 + r_k

			//////////////////////////////////////////////////////////////////////////

			m_sparseSystem->Multiply(ptrAp, ptrP, dim);								// Ap_k = A * p_k

			//////////////////////////////////////////////////////////////////////////

			float pap = Utility::VectorDot(ptrP, ptrAp, dim);							// p_k^T * A * p_k

			float alpha = rDot0 / pap;													// alpha_k = rDot_k / (p_k^T * A * p_k) 

			Utility::VectorSelfAdd(ans, ptrP, 1.f, alpha, dim);						// x_k+1 = x_k + alpha_k * p_k

			Utility::VectorSelfAdd(ptrResidual, ptrAp, 1.f, -alpha, dim);				// r_k+1 = r_k - alpha_k * Ap_k
		}

		//printf("Iteration: %d; Residual Square Norm: %f\n", it, rDot0);
	}

	void SolvePreconditioned(SeVec3f* ans, const SeVec3f* rhs, int dim, int maxIter, float tolerance, bool useTolerance)
	{
		float pap = 0.f;
		float rzDot0 = 0.f;
		float rzDot = 0.f;
		float rDot = 0.f;

		int it = 0;

		m_sparseSystem->Residual(ptrResidual, ans, rhs, dim);					// r_0 = b - A * x_0

		while (it++ < maxIter || useTolerance)
		{
			rDot = Utility::VectorDot(ptrResidual, ptrResidual, dim);			// r_k+1^T * r_k+1

			printf("It: %d; Residual Square Norm: %f\n", it - 1, rDot);

			if (rDot < tolerance) { break; }

			//////////////////////////////////////////////////////////////////////////

			m_preconditioner->Preconditioning(ptrZ, ptrResidual, dim);							// z_k = M^-1 * r_k

			//////////////////////////////////////////////////////////////////////////

			rzDot = Utility::VectorDot(ptrResidual, ptrZ, dim);				// rzDot_k = r_k^T * z_k

			// Polak-Ribiere
			float zAp = Utility::VectorDot(ptrAp, ptrZ, dim);					// zAp_k = (r_k^T - r_k-1^T) * z_k
			float beta = -zAp / pap;
			if (pap == 0.f) { beta = 0.f; }

			// Fletcher-Reeves
			//float beta = rzDot / rzDot0;										// beta_k = rzDot_k / rzDot_k-1
			//if (rzDot0 == 0.f) { beta = 0.f; }

			rzDot0 = rzDot;

			Utility::VectorSelfAdd(ptrP, ptrZ, beta, 1.f, dim);				// p_k = beta_k * p_k-1 + z_k

			//////////////////////////////////////////////////////////////////////////

			m_sparseSystem->Multiply(ptrAp, ptrP, dim);						// Ap_k = A * p_k

			//////////////////////////////////////////////////////////////////////////

			pap = Utility::VectorDot(ptrP, ptrAp, dim);						// p_k^T * Ap_k

			float alpha = rzDot / pap;											// alpha_k = rzDot_k / (p_k^T * q_k) 


			Utility::VectorSelfAdd(ans, ptrP, 1.f, alpha, dim);		 		// x_k+1 = x_k + alpha_k * p_k

			Utility::VectorSelfAdd(ptrResidual, ptrAp, 1.f, -alpha, dim);		// r_k+1 = r_k - alpha_k * Ap_k
		}
	}
};

SE_NAMESPACE_END