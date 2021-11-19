package mlp.matrix;

import mlp.activationfunction.ActivationFunction;

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

    public double l2norm() {
        double sum = 0;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                sum += data[i][j] * data[i][j];

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


    /*
     * copy a matrix
     */
    public static Matrix c(Matrix m) {
        return new Matrix(m);
    }


    /*
     * static matrix options return copies
     */

    public static Matrix transpose(Matrix m) {
        Matrix temp = new Matrix(m.cols, m.rows);
        for (int i = 0; i < m.rows; i++)
            for (int j = 0; j < m.cols; j++)
                temp.data[j][i] = m.data[i][j];

        return temp;
    }

    public static Matrix dot(Matrix a, Matrix b) {
        if (a.cols != b.rows)
            throw new ShapeMismatchException("dot shape mismatch: %s and %s\n", a.shapeString(), b.shapeString());


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

    /*
     * this is needed to be sure that the result of all mathematical operations is a valid double
     */
    private static double validMultiply(double a, double b) {
        return verifyDouble(a * b);
    }

    private static double validAddition(double a, double b) {
        return verifyDouble(a + b);
    }

    private static double verifyDouble(double o) {
        if(Double.isNaN(o))
            return 0.;
        else if(o == Double.POSITIVE_INFINITY)
            return ABSURDLY_LARGE;
        else if(o == Double.NEGATIVE_INFINITY)
            return -ABSURDLY_LARGE;
        else
            return o;
    }

    private static double verifyDouble2(double o) {
        return o != o || Double.isInfinite(o) ? 0. : o;
    }

    /*
     * functions for creating from array
     */
    public static Matrix fromArray(double[] x) {
        Matrix temp = new Matrix(x.length, 1);
        for (int i = 0; i < x.length; i++)
            temp.data[i][0] = x[i];
        return temp;

    }

    public double[] toArray() {
        return ArrayUtils.flatten(data);
    }

    /*
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
        System.out.println(this);
    }
}
