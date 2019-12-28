package goptimization

import (
	"fmt"
	"math"

	"github.com/pkg/errors"
	"gonum.org/v1/gonum/mat"
)

func Simplex(z, A, b *mat.Dense, maxIter int) (int, *mat.Dense, float64, error) {
	totalIter := 0
	cf := CanonicalForm{}
	err := cf.New(z, A, b)
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

	remap map[int]int
}

//New Initialize all the parameters in order to run the simplex algorithm
func (cf *CanonicalForm) New(z, A, b *mat.Dense) error {

	cf.remap = make(map[int]int)

	r, c := z.Dims()
	if r > 1 {
		return errors.New("z dims.r > 1")
	}
	cf.n = c

	r, c = A.Dims()
	if c > cf.n {
		return errors.New("A dims.c > z dims.r")
	}
	cf.m = r

	cf.A = A
	cf.A = cf.A.Grow(0, cf.m).(*mat.Dense)
	for i := 0; i < cf.m; i++ {
		cf.A.Set(i, cf.n+i, 1.0)
	}
	cf.b = b
	cf.c = z

	cf.c = cf.c.Grow(0, cf.m).(*mat.Dense)

	cf.x = mat.NewDense(cf.n+cf.m, 1, nil)

	cf.xBStar = mat.DenseCopyOf(b)

	cf.xN = cf.x.Slice(0, cf.n, 0, 1).(*mat.Dense)

	cf.B = cf.A.Slice(0, cf.m, cf.n, cf.n+cf.m).(*mat.Dense)

	cf.AN = cf.A.Slice(0, cf.m, 0, cf.n).(*mat.Dense)

	cf.cB = cf.c.Slice(0, 1, cf.n, cf.n+cf.m).(*mat.Dense)
	cf.cN = cf.c.Slice(0, 1, 0, cf.n).(*mat.Dense)

	j := 0
	for i := cf.n; i < cf.n+cf.m; i++ {
		cf.remap[i] = j
		j++
	}

	return nil
}

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

func (cf *CanonicalForm) FindInputVariable(y *mat.Dense, forceInputVarIndex int) (int, error) {
	var m mat.Dense
	m.Mul(y, cf.AN)
	m.Sub(cf.cN, &m)

	fmt.Printf("m:\n %v\n\n", mat.Formatted(&m, mat.Prefix(" "), mat.Excerpt(8)))
	_, c := m.Dims()

	inputVarIndex := -1
	if forceInputVarIndex != 0 {
		if m.At(0, forceInputVarIndex) > 0 {
			inputVarIndex = forceInputVarIndex
		}
	} else {
		max := 0.0
		for j := 0; j < c; j++ {
			if m.At(0, j) > 0 && m.At(0, j) > max {
				max = m.At(0, j)
				inputVarIndex = j
			}
		}
	}
	fmt.Println("inputVarIndex", inputVarIndex)
	return inputVarIndex, nil
}

func (cf *CanonicalForm) SolveBd(inputVarIndex int) (*mat.Dense, error) {
	var d mat.Dense
	err := d.Solve(cf.B, cf.AN.ColView(inputVarIndex))
	if err != nil {
		return nil, err
	}

	fmt.Printf("d:\n %v\n\n", mat.Formatted(&d, mat.Prefix(" "), mat.Excerpt(8)))

	return &d, nil
}

func (cf *CanonicalForm) FindOutputVariable(d *mat.Dense) (float64, int, error) {
	r, _ := d.Dims()
	x := math.Inf(1)
	outputVarIndex := -1

	found := false

	for i := 0; i < r; i++ {
		if d.At(i, 0) <= 0 {
			continue
		}
		found = true
		tmp := cf.xBStar.At(i, 0) / d.At(i, 0)
		fmt.Println("xOutput:", i, tmp)
		if tmp < x {
			x = tmp
			outputVarIndex = i
		}
	}
	if !found {
		return -1.0, -1, nil
	}
	fmt.Println("x", x)
	fmt.Println("outputVarIndex", outputVarIndex)
	delete(cf.remap, cf.n+outputVarIndex)
	return x, outputVarIndex, nil
}

func (cf *CanonicalForm) Update(d, y *mat.Dense, x float64, inputVarIndex, outputVarIndex int) error {
	var tmp mat.Dense
	tmp.Scale(x, d)
	cf.xBStar.Sub(cf.xBStar, &tmp)
	cf.xBStar.Set(outputVarIndex, 0, x)

	fmt.Printf("xBStar:\n %v\n\n", mat.Formatted(cf.xBStar, mat.Prefix(" "), mat.Excerpt(8)))

	r, _ := d.Dims()

	fmt.Printf("AN:\n %v\n\n", mat.Formatted(cf.AN, mat.Prefix(" "), mat.Excerpt(8)))

	for i := 0; i < r; i++ {
		cf.B.Set(i, outputVarIndex, cf.AN.At(i, inputVarIndex))
	}
	fmt.Printf("B:\n %v\n\n", mat.Formatted(cf.B, mat.Prefix(" "), mat.Excerpt(8)))

	cf.cB.Set(0, outputVarIndex, cf.cN.At(0, inputVarIndex))

	fmt.Printf("cB:\n %v\n\n", mat.Formatted(cf.cB, mat.Prefix(" "), mat.Excerpt(8)))

	return nil
}

func (cf *CanonicalForm) Iter(forceInputVarIndex int) (bool, error) {
	y, err := cf.FindY()
	if err != nil {
		return false, err
	}

	inputVarIndex, err := cf.FindInputVariable(y, forceInputVarIndex)
	if err != nil {
		return false, err
	}

	if inputVarIndex == -1 {
		return true, nil
	}

	d, err := cf.SolveBd(inputVarIndex)
	if err != nil {
		return false, err
	}
	x, outputVarIndex, err := cf.FindOutputVariable(d)
	if err != nil {
		return false, err
	}

	if outputVarIndex == -1 {
		return true, nil
	}
	cf.remap[inputVarIndex] = outputVarIndex
	err = cf.Update(d, y, x, inputVarIndex, outputVarIndex)
	if err != nil {
		return false, err
	}

	return false, nil
}

func (cf *CanonicalForm) GetResults() (*mat.Dense, float64) {
	total := float64(0)

	r, c := cf.x.Dims()

	result := mat.NewDense(r, c, nil)
	result.Zero()

	fmt.Printf("cf.cN:\n %v\n\n", mat.Formatted(cf.cN, mat.Prefix(" "), mat.Excerpt(8)))

	for xInput := 0; xInput < r; xInput++ {
		if xOutput, ok := cf.remap[xInput]; ok {
			result.Set(xInput, 0, cf.xBStar.At(xOutput, 0))
			if xInput < cf.n {
				fmt.Println(xOutput, xInput)
				total += cf.xBStar.At(xOutput, 0) * cf.cN.At(0, xInput)
			}
		}
	}

	fmt.Printf("result:\n %v\n\n", mat.Formatted(result, mat.Prefix(" "), mat.Excerpt(8)))
	fmt.Println("Score:", total)
	return result, total
}
