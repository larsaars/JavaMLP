package mlp;

import mlp.activationfunction.ActivationFunction;
import mlp.matrix.Matrix;
import mlp.utils.Log;
import mlp.utils.Serializer;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;


/**
 * MLP - short for 'Multi Layer Perceptron'
 * or also called Vanilla Neural Network
 */
public class MLP implements Serializable {

    /**
     * the layer structure:
     * weight matrices are used
     * each row is one neuron of the layer, with one belonging weight column vector (n x 1 matrix)
     * m - number of columns (number of input neurons)
     * n - number of rows (number of output neurons)
     * when taking the dot product with a column input vector, another column vector is calculated
     * with length of n - output of the layer:
     * n x m * n x 1 = n x 1
     * the result of this is passed through the activationFunction (item-wise)
     * in backprop, this whole process is done with the part-derivatives and the chain rule the other way
     * round than when forward propagating.
     */
    public Matrix[] weights, biases;

    /**
     * activation function for all layers
     */
    public ActivationFunction activationFunction;
    /**
     * the layer structure
     * (count of how many neurons are in the end in every layer)
     */
    public int[] layers;

    /**
     * the learning rate - backprop hyperparameter
     */
    public double learningRate;

    /**
     * instance of random class
     */
    private final Random random = new Random();


    public MLP(int[] layers, double learningRate, ActivationFunction activationFunction) {
        this.learningRate = learningRate;
        this.layers = layers;
        this.activationFunction = activationFunction;

        int layersSize = layers.length;

        weights = new Matrix[layersSize - 1];
        biases = new Matrix[layersSize - 1];

        for (int i = 0; i < layersSize - 1; i++) {
            weights[i] = new Matrix(layers[i + 1], layers[i], true);
            biases[i] = new Matrix(layers[i + 1], 1, true);
        }
    }

    /**
     * load neural network from file
     *
     * @param file path to file
     * @return neural network
     */
    public static MLP load(String file) {
        return (MLP) Serializer.deserialize(file);
    }


    /**
     * feed forward
     *
     * @param X input
     * @return output of last layer
     */
    public double[] predict(double[] X) {
        Matrix input = Matrix
                .columnVector(X);

        for (int i = 0; i < weights.length; i++)
            input = Matrix.dot(weights[i], input)
                    .add(biases[i])
                    .apply(activationFunction, false);

        return input.flatten();
    }


    /**
     * train in n epochs with stochastic gradient descent
     *
     * @param X      array of input samples
     * @param Y      array of expected targets
     * @param epochs number of epochs
     * @return loss
     */
    public double[] fit(double[][] X, double[][] Y, int epochs) {
        double[] loss = new double[epochs];
        for (int i = 0; i < epochs; i++) {
            int sampleN = random.nextInt(X.length);
            loss[i] = backprop(X[sampleN], Y[sampleN]);
            Log.l("epochs: " + (i + 1) + " loss: " + loss[i]);
        }

        return loss;
    }

    /**
     * train until loss is under specific loss for at least n epochs (stochastic gradient descent)
     *
     * @param X             array of input samples
     * @param Y             array of expected targets
     * @param lossTolerance loss tolerance
     * @param maxEpochs     max number of epochs
     * @return loss
     */
    public double[] fit(double[][] X, double[][] Y, double lossTolerance, int maxEpochs) {
        List<Double> loss = new ArrayList<>();

        int epochs = 0;
        double lastLoss = Double.MAX_VALUE;
        do {
            int sampleN = random.nextInt(X.length);
            double currentLoss = backprop(X[sampleN], Y[sampleN]);
            loss.add(currentLoss);

            if (Math.abs(lastLoss - currentLoss) < lossTolerance)
                break;

            lastLoss = currentLoss;
            epochs++;

            Log.l("epochs: " + epochs + " loss: " + currentLoss);
        } while (maxEpochs == -1 || epochs < maxEpochs);

        double[] lossArr = new double[loss.size()];
        for (int i = 0; i < loss.size(); i++)
            lossArr[i] = loss.get(i);

        return lossArr;
    }

    /**
     * backpropagation
     *
     * @param X array of input samples
     * @param Y array of target class
     * @return loss of the function
     */

    // TODO how to minibatch
    // TODO and how to use augmented weight vector
    private double backprop(double[] X, double[] Y) {
        Matrix[] feedforward = new Matrix[layers.length];
        feedforward[0] = Matrix.columnVector(X);

        for (int i = 0; i < feedforward.length - 1; i++)
            feedforward[i + 1] = Matrix.dot(weights[i], feedforward[i])
                    .add(biases[i])
                    .apply(activationFunction, false);

        Matrix inputWeightsBefore = Matrix.columnVector(X);
        // expected minus wished output
        Matrix errorBefore = Matrix.columnVector(Y)
                .subtract(feedforward[feedforward.length - 1]);

        double loss = errorBefore.r2error();

        for (int i = layers.length - 2; i >= 0; i--) {
            Matrix error = i == layers.length - 2 ? errorBefore : Matrix.dot(inputWeightsBefore, errorBefore); // z

            Matrix gradient = Matrix.c(feedforward[i + 1])
                    .apply(activationFunction, true)
                    .multiply(error)
                    .multiply(learningRate); // w * g'(z)

            Matrix l_T = Matrix.transpose(feedforward[i]);
            Matrix l_delta = Matrix.dot(gradient, l_T);

            weights[i].add(l_delta);
            biases[i].add(gradient);

            if (i != 0) {
                inputWeightsBefore = Matrix.transpose(weights[i]);
                errorBefore = error;
            }
        }

        return loss;
    }

    public void printNetwork() {
        for (int i = 0; i < layers.length - 1; i++) {
            System.out.println("Layer " + i);
            System.out.println("Weights: " + weights[i]);
            System.out.println("Biases: " + biases[i]);
        }
    }
}

