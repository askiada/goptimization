package goptimization

import (
	"fmt"
	"math"

	"github.com/pkg/errors"
	"gonum.org/v1/gonum/mat"
)

// Simplex Solve a linear problem wihtout strict inequality constraints.
// Input follows standard form:
// Maximize z = Σ(1<=j<=n) c_j*x_j
// Constraints:
// 1<=i<=m,  Σ(1<=j<=n) a_i_j*x_j <= b_i
// 1<=j<=n x_j >= 0
// - Define the canonical form of the problem (add slack variables and transfrom inequality constraints to equality constraints)
// - Check the basic solution is feasible, if not you need to run two phases simplex
// - Run iterations
// - Stop when the optimal solution is found or after maxIter
// Apply
// - First Danzig critera: for entering variable, pick the nonbasic variable with the largest reduced cost.
// - Bland's rule to avoid cycles : Choose the entering basic variable xj such that j is the smallest
// index with c¯j < 0. Also choose the leaving basic variable i with the smallest index (in case of ties in the ratio test)
func Simplex(c, A, b *mat.Dense, maxIter int) (int, *mat.Dense, float64, error) {
	totalIter := 0
	cf := CanonicalForm{}
	err := cf.New(c, A, b)
	if err != nil {
		return 0, nil, 0, err
	}
	for i := 0; i < maxIter; i++ {
		end, err := cf.Iter(0)
		if err != nil {
			return 0, nil, 0, err
		}
		if end {
			break
		}
		totalIter++
	}
	results, score := cf.GetResults()
	return totalIter, results, score, nil
}

// CanonicalForm Canonical form of a linear optimizattion problem
type CanonicalForm struct {
	// Positivity constraints
	n int
	// Equality constraints
	m int
	// Matrix (m, n+m)
	A *mat.Dense
	// Column vector (n+m)
	x *mat.Dense
	// Row vector (n+m)
	c *mat.Dense
	// Column vector (m)
	b *mat.Dense

	//Feasible solutions in the current dictionary
	// Column Vector (m)
	xBStar *mat.Dense
	// Column Vector (n)
	xN *mat.Dense

	// Matrix (m, m)
	B *mat.Dense
	// Matrix (m,n)
	AN *mat.Dense

	//Row Vector (m)
	cB *mat.Dense
	//Row Vector (n)
	cN *mat.Dense

	remap []int
}

//New Initialize all the parameters in order to run the simplex algorithm
func (cf *CanonicalForm) New(c, A, b *mat.Dense) error {

	rows, cols := c.Dims()
	if rows > 1 {
		return errors.New("z dims.r > 1")
	}
	cf.n = cols

	rows, cols = A.Dims()
	if cols > cf.n {
		return errors.New("A dims.c > z dims.r")
	}
	cf.m = rows

	cf.A = A
	cf.A = cf.A.Grow(0, cf.m).(*mat.Dense)
	//Add slack variables
	for i := 0; i < cf.m; i++ {
		cf.A.Set(i, cf.n+i, 1.0)
	}
	cf.b = b
	cf.c = c

	cf.c = cf.c.Grow(0, cf.m).(*mat.Dense)

	cf.x = mat.NewDense(cf.n+cf.m, 1, nil)

	cf.xBStar = mat.DenseCopyOf(b)

	cf.xN = cf.x.Slice(0, cf.n, 0, 1).(*mat.Dense)

	cf.B = cf.A.Slice(0, cf.m, cf.n, cf.n+cf.m).(*mat.Dense)

	cf.AN = cf.A.Slice(0, cf.m, 0, cf.n).(*mat.Dense)

	cf.cB = cf.c.Slice(0, 1, cf.n, cf.n+cf.m).(*mat.Dense)
	cf.cN = cf.c.Slice(0, 1, 0, cf.n).(*mat.Dense)

	//Store the entring and leaving pairs for each iteration
	cf.remap = make([]int, cf.n+cf.m)
	for i := 0; i < cf.n+cf.m; i++ {
		cf.remap[i] = i
	}
	return nil
}

//FindY Extract and solve a sub problem of the current dictionary
// The current dictionary is:
// (1) xB = xBStar - B^-1*AN*xN
// (2) z = zStar + (cN - cB*B^-1*AN)xN
// Set y=cB*B^-1 and solve it
func (cf *CanonicalForm) FindY() (*mat.Dense, error) {

	var y, BInv mat.Dense
	err := BInv.Inverse(cf.B)
	if err != nil {
		return nil, err
	}
	y.Mul(cf.cB, &BInv)

	fmt.Printf("y:\n %v\n\n", mat.Formatted(&y, mat.Prefix(" "), mat.Excerpt(8)))

	return &y, nil
}

//FindEnteringVariable Define the best entering varialbe following Danzig criteria and Bland's rule
// Find one column a^k of A not in B with y*a^k<c^k
// If there is no entering column, the current solution is optimal
func (cf *CanonicalForm) FindEnteringVariable(y *mat.Dense, forceEnteringVarIndex int) (int, error) {
	var m mat.Dense
	m.Mul(y, cf.AN)
	m.Sub(cf.cN, &m)

	_, c := m.Dims()

	enteringVarIndex := -1
	if forceEnteringVarIndex != 0 {
		if m.At(0, forceEnteringVarIndex) > 0 {
			enteringVarIndex = forceEnteringVarIndex
		}
	} else {
		max := 0.0
		for j := 0; j < c; j++ {
			//First Danzig criteria and Bland's rule
			if m.At(0, j) > 0 && m.At(0, j) > max {
				max = m.At(0, j)
				enteringVarIndex = j
			}
		}
	}
	fmt.Println("enteringVarIndex", enteringVarIndex)
	return enteringVarIndex, nil
}

//SolveBd FindY describes the current dictionary.
// To find the best leaving variable we start from (1) and set d=B^-1*a^k
// When we solve it, we get the equation to maximize in order to find the leaving variable
func (cf *CanonicalForm) SolveBd(enteringVarIndex int) (*mat.Dense, error) {
	var d mat.Dense
	err := d.Solve(cf.B, cf.AN.ColView(enteringVarIndex))
	if err != nil {
		return nil, err
	}

	fmt.Printf("d:\n %v\n\n", mat.Formatted(&d, mat.Prefix(" "), mat.Excerpt(8)))

	return &d, nil
}

// FindLeavingVariable Define what is the best leaving variable following Bland's rule
// Find the biggest x_kStar with x_BStar - x_kStar*d >= 0
// If d<=0, the algorithm ends and the problem is unbounded.
// Otherwise, the biggest x_kStar force one of the components of x_BStar - x_kStar*d to be equal to zero
// and defines the leaving variable
func (cf *CanonicalForm) FindLeavingVariable(d *mat.Dense) (float64, int, error) {
	r, _ := d.Dims()
	x := math.Inf(1)
	leavingVarIndex := -1

	found := false

	for i := 0; i < r; i++ {
		if d.At(i, 0) <= 0 {
			continue
		}
		found = true
		tmp := cf.xBStar.At(i, 0) / d.At(i, 0)
		fmt.Println("xLeaving:", i, tmp)
		//Bland's rule
		if tmp < x {
			x = tmp
			leavingVarIndex = i
		}
	}
	if !found {
		return -1.0, -1, nil
	}
	fmt.Println("x", x)
	fmt.Println("leavingVarIndex", leavingVarIndex)
	return x, leavingVarIndex, nil
}

// Update Update the dictionary in order to run anotheriteration
// Replace the leaving variable with the entering variable in xBStar
// Replace the leaving column in the base B with the entering column
func (cf *CanonicalForm) Update(d, y *mat.Dense, x float64, enteringVarIndex, leavingVarIndex int) error {
	var tmp mat.Dense
	tmp.Scale(x, d)
	cf.xBStar.Sub(cf.xBStar, &tmp)
	cf.xBStar.Set(leavingVarIndex, 0, x)

	fmt.Printf("xBStar:\n %v\n\n", mat.Formatted(cf.xBStar, mat.Prefix(" "), mat.Excerpt(8)))

	r, _ := d.Dims()

	leavingCol := mat.DenseCopyOf(cf.B.ColView(leavingVarIndex))
	leavingC := cf.cB.At(0, leavingVarIndex)
	for i := 0; i < r; i++ {
		cf.B.Set(i, leavingVarIndex, cf.AN.At(i, enteringVarIndex))
		cf.AN.Set(i, enteringVarIndex, leavingCol.At(i, 0))
	}
	cf.cB.Set(0, leavingVarIndex, cf.cN.At(0, enteringVarIndex))
	cf.cN.Set(0, enteringVarIndex, leavingC)

	fmt.Printf("A:\n %v\n\n", mat.Formatted(cf.A, mat.Prefix(" "), mat.Excerpt(8)))

	fmt.Printf("cB:\n %v\n\n", mat.Formatted(cf.cB, mat.Prefix(" "), mat.Excerpt(8)))

	return nil
}

//Iter Run one iteration of the simplex algorithm
func (cf *CanonicalForm) Iter(forceEnteringVarIndex int) (bool, error) {
	//Solve yB=c_B
	y, err := cf.FindY()
	if err != nil {
		return false, err
	}
	//Find a entering column/variable
	enteringVarIndex, err := cf.FindEnteringVariable(y, forceEnteringVarIndex)
	if err != nil {
		return false, err
	}

	// The algorithm ends when there is no candidates
	if enteringVarIndex == -1 {
		return true, nil
	}

	//Solve Bd=a^k
	d, err := cf.SolveBd(enteringVarIndex)
	if err != nil {
		return false, err
	}

	// Find the leaving column/variable
	x, leavingVarIndex, err := cf.FindLeavingVariable(d)
	if err != nil {
		return false, err
	}
	// The algorithm ends when there is no candidates
	if leavingVarIndex == -1 {
		return true, nil
	}

	//Store the new pair of entering/leaving variables
	tmp := cf.remap[cf.n+leavingVarIndex]
	cf.remap[cf.n+leavingVarIndex] = enteringVarIndex
	cf.remap[enteringVarIndex] = tmp

	// Update the dictionary for the next iteration
	err = cf.Update(d, y, x, enteringVarIndex, leavingVarIndex)
	if err != nil {
		return false, err
	}

	return false, nil
}

// GetResults Build the solution.
// It returns a matrix (n+m,1), the first n components are the best value for the problem and the others are the "leftover" for each constraint.
// Also returns the maximum score.
func (cf *CanonicalForm) GetResults() (*mat.Dense, float64) {
	total := float64(0)

	rows, cols := cf.x.Dims()

	result := mat.NewDense(rows, cols, nil)
	result.Zero()

	for i := cf.n; i < cf.n+cf.m; i++ {
		result.Set(cf.remap[i], 0, cf.xBStar.At(i-cf.n, 0))
		if cf.remap[i] < cf.n {
			total += cf.xBStar.At(i-cf.n, 0) * cf.c.At(0, i)
		}
	}

	fmt.Printf("result:\n %v\n\n", mat.Formatted(result, mat.Prefix(" "), mat.Excerpt(8)))
	fmt.Println("Score:", total)
	return result, total
}
