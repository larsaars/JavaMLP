package mlp;

import mlp.activationfunction.ActivationFunction;
import mlp.matrix.Matrix;
import mlp.utils.Log;
import mlp.utils.Serializer;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;


public class NeuralNetwork implements Serializable {

    public Matrix[] weights, biases;
    public ActivationFunction[] activationFunctions;
    public int[] layers;

    public double learningRate;

    private final Random random = new Random();


    public NeuralNetwork(int[] layers, double learningRate, ActivationFunction activationFunction) {
        this.learningRate = learningRate;
        this.layers = layers;
        int layersSize = layers.length;

        weights = new Matrix[layersSize - 1];
        biases = new Matrix[layersSize - 1];
        activationFunctions = new ActivationFunction[layersSize - 1];

        for (int i = 0; i < layersSize - 1; i++) {
            activationFunctions[i] = activationFunction;
            weights[i] = new Matrix(layers[i + 1], layers[i], true);
            biases[i] = new Matrix(layers[i + 1], 1, true);
        }
    }

    /**
     * load neural network from file
     * @param file path to file
     * @return neural network
     */
    public static NeuralNetwork load(String file) {
        return (NeuralNetwork) Serializer.deserialize(file);
    }


    /**
     * feed forward
     * @param X input
     * @return output of last layer
     */
    public double[] predict(double[] X) {
        Matrix input = Matrix.fromArray(X);

        for (int i = 0; i < weights.length; i++)
            input = Matrix.dot(weights[i], input)
                    .add(biases[i])
                    .apply(activationFunctions[i], false);

        return input.toArray();
    }


    /**
     * train in n epochs with stochastic gradient descent
     * @param X array of input samples
     * @param Y array of expected targets
     * @param epochs number of epochs
     * @return loss
     */
    public double[] fit(double[][] X, double[][] Y, int epochs) {
        double[] loss = new double[epochs];
        for (int i = 0; i < epochs; i++) {
            int sampleN = random.nextInt(X.length);
            loss[i] = backprop(X[sampleN], Y[sampleN]);
        }

        return loss;
    }

    /**
     * train until loss is under specific loss for at least n epochs (stochastic gradient descent)
     * @param X array of input samples
     * @param Y array of expected targets
     * @param minLoss minimum loss to stop training
     * @param forAtLeastN epochs
     * @param maxEpochs maximum epochs (put -1 for infinite)
     * @return loss
     */
    public double[] fit(double[][] X, double[][] Y, double minLoss, int forAtLeastN, int maxEpochs) {
        List<Double> loss = new ArrayList<>();

        int epochsUnderMinLoss = 0, epochs = 0;
        do {
            int sampleN = random.nextInt(X.length);
            double currentLoss = backprop(X[sampleN], Y[sampleN]);
            loss.add(currentLoss);

            if (currentLoss <= minLoss)
                epochsUnderMinLoss++;
            else
                epochsUnderMinLoss = 0;

            epochs++;

            Log.l("epochs: " + epochs + " loss: " + currentLoss);
        } while (epochsUnderMinLoss < forAtLeastN && (maxEpochs == -1 || epochs < maxEpochs));

        double[] lossArr = new double[loss.size()];
        for(int i = 0; i < loss.size(); i++)
            lossArr[i] = loss.get(i);

        return lossArr;
    }

    /**
     * backpropagation
     * @param X array of input samples
     * @param Y array of target class
     * @return loss of the function
     */
    private double backprop(double[] X, double[] Y) {
        Matrix[] processing = new Matrix[layers.length];
        processing[0] = Matrix.fromArray(X);

        for(int i = 0; i < processing.length - 1; i++)
            processing[i + 1] = Matrix.dot(weights[i], processing[i])
                    .add(biases[i])
                    .apply(activationFunctions[i], false);

        Matrix inputWeightsBefore = Matrix.fromArray(X);
        // expected minus wished output
        Matrix errorBefore = Matrix.fromArray(Y)
                .subtract(processing[processing.length - 1]);
        double loss = errorBefore.l2norm();

        for (int i = layers.length - 2; i >= 0; i--) {
            Matrix error = i == layers.length - 2 ? errorBefore : Matrix.dot(inputWeightsBefore, errorBefore);

            Matrix gradient = Matrix.c(processing[i + 1])
                    .apply(activationFunctions[i], true)
                    .multiply(error)
                    .multiply(learningRate);

            Matrix l_T = Matrix.transpose(processing[i]);
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

