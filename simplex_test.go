package goptimization

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/mat"
)

func TestSimplex(t *testing.T) {
	z := mat.NewDense(1, 4, []float64{7, 9, 18, 17})
	A := mat.NewDense(3, 4, []float64{
		2, 4, 5, 7,
		1, 1, 2, 2,
		1, 2, 3, 3,
	})
	b := mat.NewDense(3, 1, []float64{42, 17, 24})

	totalIter, results, score, err := Simplex(z, A, b, 10)
	require.NoError(t, err)
	assert.Equal(t, 2, totalIter)
	assert.True(t, mat.EqualApprox(mat.NewDense(7, 1, []float64{3, 0, 7, 0, 1, 0, 0}), results, 0.000001))
	assert.Equal(t, 147.0, score)
}

func TestSimplex2(t *testing.T) {
	z := mat.NewDense(1, 2, []float64{100, 85})
	A := mat.NewDense(3, 2, []float64{
		12, 24,
		9, 5,
		30, 30,
	})
	b := mat.NewDense(3, 1, []float64{480, 180, 720})

	totalIter, results, score, err := Simplex(z, A, b, 10)
	require.NoError(t, err)
	assert.Equal(t, 2, totalIter)
	assert.True(t, mat.EqualApprox(mat.NewDense(5, 1, []float64{15, 9, 84, 0, 0}), results, 0.000001))
	assert.Equal(t, 2265.0, score)
}

func TestSimplex3(t *testing.T) {
	z := mat.NewDense(1, 3, []float64{4, 3, 5})
	A := mat.NewDense(3, 3, []float64{
		4, 12, 8,
		4, 4, 8,
		12, 4, 8,
	})
	b := mat.NewDense(3, 1, []float64{4800, 4000, 5600})

	totalIter, results, score, err := Simplex(z, A, b, 10)
	require.NoError(t, err)
	assert.Equal(t, 3, totalIter)
	assert.True(t, mat.EqualApprox(mat.NewDense(6, 1, []float64{200, 100, 350, 0, 0, 0}), results, 0.000001))
	assert.Equal(t, 2850.0, score)
}

func TestIter(t *testing.T) {
	z := mat.NewDense(1, 4, []float64{7, 9, 18, 17})
	AConstraints := mat.NewDense(3, 4, []float64{
		2, 4, 5, 7,
		1, 1, 2, 2,
		1, 2, 3, 3,
	})
	b := mat.NewDense(3, 1, []float64{42, 17, 24})

	cf := CanonicalForm{}
	err := cf.New(z, AConstraints, b)
	require.NoError(t, err)

	end, err := cf.Iter(2)
	require.NoError(t, err)
	require.False(t, end)
	assert.True(t, mat.Equal(mat.NewDense(3, 7, []float64{
		2, 4, 5, 7, 1, 0, 5,
		1, 1, 2, 2, 0, 1, 2,
		1, 2, 3, 3, 0, 0, 3,
	}), cf.A))

	assert.True(t, mat.Equal(b, cf.b))
	assert.True(t, mat.Equal(mat.NewDense(1, 7, []float64{7, 9, 18, 17, 0, 0, 18}), cf.c))
	assert.True(t, mat.Equal(mat.NewDense(3, 1, []float64{2, 1, 8}), cf.xBStar))
	assert.True(t, mat.Equal(mat.NewDense(3, 3, []float64{1, 0, 5, 0, 1, 2, 0, 0, 3}), cf.B))
	assert.True(t, mat.Equal(AConstraints, cf.AN))

	end, err = cf.Iter(0)
	require.NoError(t, err)
	require.False(t, end)
	assert.True(t, mat.Equal(mat.NewDense(3, 7, []float64{
		2, 4, 5, 7, 1, 2, 5,
		1, 1, 2, 2, 0, 1, 2,
		1, 2, 3, 3, 0, 1, 3,
	}), cf.A))

	assert.True(t, mat.Equal(b, cf.b))
	assert.True(t, mat.Equal(mat.NewDense(1, 7, []float64{7, 9, 18, 17, 0, 7, 18}), cf.c))
	assert.True(t, mat.EqualApprox(mat.NewDense(3, 1, []float64{1, 3, 7}), cf.xBStar, 0.000001))
	assert.True(t, mat.Equal(mat.NewDense(3, 3, []float64{1, 2, 5, 0, 1, 2, 0, 1, 3}), cf.B))
	assert.True(t, mat.Equal(AConstraints, cf.AN))

	end, err = cf.Iter(0)
	require.NoError(t, err)
	require.True(t, end)
	results, score := cf.GetResults()
	assert.True(t, mat.EqualApprox(mat.NewDense(7, 1, []float64{3, 0, 7, 0, 1, 0, 0}), results, 0.000001))
	assert.Equal(t, 147.0, score)

	end, err = cf.Iter(0)
	require.NoError(t, err)
	require.True(t, end)
	results, score = cf.GetResults()
	assert.True(t, mat.EqualApprox(mat.NewDense(7, 1, []float64{3, 0, 7, 0, 1, 0, 0}), results, 0.000001))
	assert.Equal(t, 147.0, score)
}

func TestNew(t *testing.T) {
	z := mat.NewDense(1, 4, []float64{7, 9, 18, 17})
	AConstraints := mat.NewDense(3, 4, []float64{
		2, 4, 5, 7,
		1, 1, 2, 2,
		1, 2, 3, 3,
	})
	b := mat.NewDense(3, 1, []float64{42, 17, 24})

	cf := CanonicalForm{}
	err := cf.New(z, AConstraints, b)
	require.NoError(t, err)

	assert.Equal(t, cf.m, 3)
	assert.Equal(t, cf.n, 4)

	A := mat.NewDense(3, 7, []float64{
		2, 4, 5, 7, 1, 0, 0,
		1, 1, 2, 2, 0, 1, 0,
		1, 2, 3, 3, 0, 0, 1,
	})
	assert.True(t, mat.Equal(A, cf.A))
	assert.True(t, mat.Equal(b, cf.b))

	assert.True(t, mat.Equal(mat.NewDense(1, 7, []float64{7, 9, 18, 17, 0, 0, 0}), cf.c))

	cf.x.SetCol(0, []float64{1, 2, 3, 4, 5, 6, 7})

	//assert.True(t, mat.Equal(mat.NewDense(3, 1, []float64{5, 6, 7}), cf.xB))

	assert.True(t, mat.Equal(mat.NewDense(4, 1, []float64{1, 2, 3, 4}), cf.xN))

	assert.True(t, mat.Equal(mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}), cf.B))

	assert.True(t, mat.Equal(AConstraints, cf.AN))

	assert.True(t, mat.Equal(mat.NewDense(1, 3, []float64{0, 0, 0}), cf.cB))

	assert.True(t, mat.Equal(mat.NewDense(1, 4, []float64{7, 9, 18, 17}), cf.cN))

	y, err := cf.FindY()
	require.NoError(t, err)
	assert.True(t, mat.Equal(mat.NewDense(1, 3, []float64{0, 0, 0}), y))

	enteringVarIndex, err := cf.FindEnteringVariable(y, 2)
	require.NoError(t, err)
	assert.Equal(t, 2, enteringVarIndex)

	d, err := cf.SolveBd(enteringVarIndex)
	require.NoError(t, err)
	assert.True(t, mat.Equal(mat.NewDense(3, 1, []float64{5, 2, 3}), d))
	x, leavingVarIndex, err := cf.FindLeavingVariable(d)
	require.NoError(t, err)
	assert.Equal(t, 2, leavingVarIndex)
	assert.Equal(t, 8.0, x)

	err = cf.Update(d, y, x, enteringVarIndex, leavingVarIndex)
	require.NoError(t, err)
	assert.True(t, mat.Equal(mat.NewDense(3, 1, []float64{2, 1, 8}), cf.xBStar))

	assert.True(t, mat.Equal(mat.NewDense(3, 3, []float64{1, 0, 5, 0, 1, 2, 0, 0, 3}), cf.B))

	assert.True(t, mat.Equal(mat.NewDense(1, 3, []float64{0, 0, 18}), cf.cB))
}

func TestInverse(t *testing.T) {
	B := mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1})
	BInv := mat.DenseCopyOf(B)
	err := B.Inverse(BInv)
	require.NoError(t, err)
}
