package mlp.matrix;

import mlp.activationfunction.ActivationFunction;
import mlp.utils.Log;

import java.io.Serializable;
import java.util.*;

public class Matrix implements Serializable {
    public static final double ABSURDLY_LARGE = 1e9;

    public double[][] data;
    public int rows, cols;

    public Matrix(int rows, int cols, boolean random) {
        this.rows = rows;
        this.cols = cols;
        data = new double[rows][cols];

        if (random) randomize();
    }

    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows][cols];
    }

    public Matrix(Matrix matrix) {
        this.rows = matrix.rows;
        this.cols = matrix.cols;

        data = Arrays.stream(matrix.data).map(double[]::clone).toArray(double[][]::new);
    }

    public Matrix(double[][] data) {
        this.rows = data.length;
        this.cols = data[0].length;
        this.data = data;
    }

    public Matrix expandByColumn(int index, double filler) {
        Matrix m = new Matrix(rows, cols + 1);

        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                if (j == index)
                    m.data[i][j] = filler;
                else
                    m.data[i][j] = data[i][j > index ? j - 1 : j];
            }
        }

        return m;
    }

    public Matrix expandByRow(int index, double filler) {
        Matrix m = new Matrix(rows + 1, cols);

        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                if (i == index)
                    m.data[i][j] = filler;
                else
                    m.data[i][j] = data[i > index ? i - 1 : i][j];
            }
        }

        return m;
    }

    public Matrix add(double scalar) {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i][j] = validAddition(data[i][j], scalar);
        // data[i][j] += scalar;

        return this;
    }

    public Matrix add(Matrix m) {
        if (cols != m.cols || rows != m.rows)
            throw new ShapeMismatchException("add shape mismatch: %s and %s\n", shapeString(), m.shapeString());


        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i][j] = validAddition(data[i][j], m.data[i][j]);
        // data[i][j] += m.data[i][j];

        return this;
    }

    public Matrix subtract(Matrix m) {
        if (cols != m.cols || rows != m.rows)
            throw new ShapeMismatchException("subtract shape mismatch: %s and %s\n", shapeString(), m.shapeString());

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                // data[i][j] -= m.data[i][j];
                data[i][j] = validAddition(data[i][j], -m.data[i][j]);

        return this;
    }


    public Matrix multiply(double scalar) {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                // data[i][j] *= scalar;
                data[i][j] = validMultiply(data[i][j], scalar);

        return this;
    }

    public Matrix multiplyExceptColumn(double scalar, int index) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (j == index) continue;
                data[i][j] = validMultiply(data[i][j], scalar);
            }
        }

        return this;
    }

    public Matrix squared() {
        return multiply(this);
    }

    public Matrix abs() {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i][j] = Math.abs(data[i][j]);

        return this;
    }

    // elementwise multiplication
    public Matrix multiply(Matrix m) {
        if (cols != m.cols || rows != m.rows)
            throw new ShapeMismatchException("multiply shape mismatch: %s and %s\n", shapeString(), m.shapeString());


        for (int i = 0; i < m.rows; i++)
            for (int j = 0; j < m.cols; j++)
                //data[i][j] *= m.data[i][j];
                data[i][j] = validMultiply(data[i][j], m.data[i][j]);

        return this;
    }


    public Matrix apply(ActivationFunction activationFunction, boolean derive) {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i][j] = derive ? activationFunction.derive(data[i][j]) : activationFunction.activate(data[i][j]);

        return this;
    }

    public Matrix dot(Matrix m) {
        Matrix temp = Matrix.dot(this, m);
        rows = temp.rows;
        cols = temp.cols;
        data = temp.data;
        return this;
    }

    public Matrix transpose() {
        Matrix temp = Matrix.transpose(this);
        rows = temp.rows;
        cols = temp.cols;
        data = temp.data;
        return this;
    }

    public double l2norm() {
        double sum = 0;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                sum += Math.pow(data[i][j], 2);

        return Math.sqrt(sum);
    }

    public double l1norm() {
        double sum = 0;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                sum += Math.abs(data[i][j]);

        return sum;
    }

    public double r2error() {
        return l2norm() / (rows * cols);
    }


    /**
     * @return copy of matrix
     */
    public static Matrix c(Matrix m) {
        return new Matrix(m);
    }

    /**
     * @return random matrix with values between -1 and 1
     */
    public static Matrix random(int rows, int cols) {
        return new Matrix(rows, cols, true);
    }

    /**
     * @return return empty matrix with zeros as filler
     */
    public static Matrix zeros(int rows, int cols) {
        return new Matrix(rows, cols);
    }


    /**
     * transpose a matrix
     *
     * @param m matrix to transpose
     * @return transposed matrix (a copy)
     */
    public static Matrix transpose(Matrix m) {
        Matrix temp = new Matrix(m.cols, m.rows);
        for (int i = 0; i < m.rows; i++)
            for (int j = 0; j < m.cols; j++)
                temp.data[j][i] = m.data[i][j];

        return temp;
    }

    /**
     * matrix multiplication
     *
     * @param a left matrix
     * @param b right matrix
     * @return a matmul b
     */
    public static Matrix dot(Matrix a, Matrix b) {
        if (a.cols != b.rows)
             throw new ShapeMismatchException("dot shape mismatch: %s and %s\nmatrix 1:\n%s\nmatrix 2:\n%s", a.shapeString(), b.shapeString(), a.toString(), b.toString());

        Matrix temp = new Matrix(a.rows, b.cols);
        for (int i = 0; i < temp.rows; i++)
            for (int j = 0; j < temp.cols; j++)
                for (int k = 0; k < a.cols; k++)
                    // temp.data[i][j] += validMultiply(a.data[i][k], b.data[k][j]);
                    temp.data[i][j] = validAddition(temp.data[i][j], validMultiply(a.data[i][k], b.data[k][j]));

        return temp;
    }

    public int[] shape() {
        return new int[]{rows, cols};
    }

    public String shapeString() {
        return Arrays.toString(shape());
    }


    // assign random values between -1 and 1 to matrix
    public void randomize() {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i][j] = Math.random() * 2. - 1.;
    }

    /**
     * this is needed to be sure that the result of all mathematical operations is a valid double
     */
    private static double validMultiply(double a, double b) {
        return verifyDouble(a * b);
    }

    private static double validAddition(double a, double b) {
        return verifyDouble(a + b);
    }

    private static double verifyDouble(double o) {
        if (Double.isNaN(o))
            return 0.;
        else if (o == Double.POSITIVE_INFINITY)
            return ABSURDLY_LARGE;
        else if (o == Double.NEGATIVE_INFINITY)
            return -ABSURDLY_LARGE;
        else
            return o;
    }

    private static double verifyDouble2(double o) {
        return o != o || Double.isInfinite(o) ? 0. : o;
    }

    /**
     * functions for creating from array
     */
    public static Matrix columnVector(double[] a) {
        Matrix m = new Matrix(a.length, 1);

        for (int i = 0; i < a.length; i++)
            m.data[i][0] = a[i];

        return m;
    }

    public double[] flatten() {
        return ArrayUtils.flatten(data);
    }

    /**
     * helpful utility functions
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                sb.append(data[i][j]).append(" ");
            }
            sb.append('\n');
        }

        return sb.toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Matrix)) return false;
        Matrix matrix = (Matrix) o;
        return rows == matrix.rows && cols == matrix.cols && Arrays.deepEquals(data, matrix.data);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(rows, cols);
        result = 31 * result + Arrays.deepHashCode(data);
        return result;
    }

    public void print() {
        System.out.println(shapeString());
        System.out.println(this);
    }

    public static void print(Matrix[] ms) {
        for(int i = 0; i < ms.length; i++) {
            System.out.println(i);
            if(ms[i] != null)
                ms[i].print();
            else
                System.out.println("Matrix is null");
        }
    }
}
