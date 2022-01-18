package mlp.matrix;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class ArrayUtils {

    public static final Random random = new Random();

    public static double[] flatten(double[][] array) {
        double[] flat = new double[array.length * array[0].length];
        for (int i = 0; i < array.length; i++)
            System.arraycopy(array[i], 0, flat, i * array[0].length, array[i].length);
        return flat;
    }

    public static double[][] unflatten(double[] flat, int rows, int cols) {
        double[][] array = new double[rows][cols];
        for (int i = 0; i < rows; i++)
            System.arraycopy(flat, i * cols, array[i], 0, cols);
        return array;
    }

    public static double[][] fromList(List<Double[]> list) {
        double[][] array = new double[list.size()][];

        for (int i = 0; i < list.size(); i++)
            array[i] = toPrimitive(list.get(i));

        return array;
    }

    public static double[] toPrimitive(Double[] array) {
        double[] result = new double[array.length];
        for (int i = 0; i < array.length; i++)
            result[i] = array[i];
        return result;
    }

    public static Double[] toObject(double[] array) {
        Double[] result = new Double[array.length];
        for (int i = 0; i < array.length; i++)
            result[i] = array[i];
        return result;
    }

    public static double[] oneHotEncoding(int size, int index) {
        double[] result = new double[size];
        result[index] = 1.;
        return result;
    }

    public static int argMax(double[] array) {
        int max = 0;
        for (int i = 1; i < array.length; i++)
            if (array[i] > array[max]) max = i;
        return max;
    }

    public static int argMin(double[] array) {
        int min = 0;
        for (int i = 1; i < array.length; i++)
            if (array[i] < array[min]) min = i;
        return min;
    }

    /**
     * shuffle multiple arrays in the same manner
     * https://stackoverflow.com/a/19333201/5899585
     */
    public static <T> void shuffle(T[]... arrays) {
        int countOfArrays = arrays[0].length;
        for (int i = 1; i < arrays.length ; i++) {
            if(arrays[i].length != countOfArrays)
                throw new IllegalArgumentException("All arrays must have the same length");
        }


        int count = arrays[0].length;
        for (int i = count; i > 1; i--) {
            int swapIdx = random.nextInt(i);

            for (T[] array : arrays)
                swap(array, i - 1, swapIdx);
        }
    }

    /**
     * swap two array elements
     */
    public static <T> void swap(T[] array, int i, int j) {
        T temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }


    /**
     * @return last element of array
     */
    public static <T> T lastElement(T[] array) {
        return array[array.length - 1];
    }
}

