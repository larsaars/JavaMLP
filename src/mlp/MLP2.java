package mlp;

import mlp.activationfunction.ActivationFunction;
import mlp.activationfunction.ActivationFunctions;
import mlp.matrix.ArrayUtils;
import mlp.matrix.Matrix;
import mlp.utils.Log;
import mlp.utils.Pair;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Simple implementation of a Multi Layer Perceptron learning with mean squared error loss function
 *
 * Hyperparameters that can be edited:
 * batch size
 * layer structure
 * learning rate for bias and for weights
 * activation functions
 *
 * Might not work sometimes, the matrix library was implemented by myself and
 * when using ReLU for example, there are often infinity / NaN errors occurring, since the numbers get too big.
 *
 * Implemented with the help of this video and my lectures on neuronal networks. Thanks to my prof BJ.
 * https://www.youtube.com/watch?v=x_Eamf8MHwU
 */
public class MLP2 implements Serializable {
    /**
     * thetas, weight matrices
     * bias vectors are not included in the weights,
     * using extra vector, easier to calculate
     */
    public Matrix[] weight, bias;

    /**
     * activation functions for all layers
     * g(x)
     *
     * the output activation function is used to activate the last layer
     */
    public ActivationFunction activationFunction, outputActivationFunction;

    /**
     * layer structure
     */
    public int[] layerStructure;

    /**
     * hyperparameter learning rate eta for weights
     * and for bias (gamma)
     */
    public double learningRate, biasLearningRate;

    /**
     * initializer
     *
     * @param layerStructure     layer structure
     * @param activationFunction activation function
     * @param learningRate       learning rate
     */
    public MLP2(int[] layerStructure, ActivationFunction activationFunction, ActivationFunction outputActivationFunction, double learningRate, double biasLearningRate) {
        this.learningRate = learningRate;
        this.biasLearningRate = biasLearningRate;
        this.layerStructure = layerStructure;
        this.activationFunction = activationFunction;
        this.outputActivationFunction = outputActivationFunction;

        // initialize weights and biases structure
        weight = new Matrix[layerStructure.length - 1];
        bias = new Matrix[layerStructure.length - 1];

        for (int i = 0; i < weight.length; i++) {
            weight[i] = Matrix.random(layerStructure[i + 1], layerStructure[i]);
            bias[i] = Matrix.random(layerStructure[i + 1], 1);
        }
    }

    /**
     * Calculate zs and activations for each layer
     * z matrix array is one less than the activations matrix array
     * z array starts at 'second element' (Z2)
     * a array starts at 'first element' (A1)
     *
     * @param input column vector of y inputs
     * @return zs and activations
     */
    public Pair<Matrix[], Matrix[]> feedForward(Matrix input) {
        Matrix[] z = new Matrix[layerStructure.length - 1], // z
                a = new Matrix[layerStructure.length]; // activations (g(z))

        // first activation is the input
        a[0] = input;

        // calculate zs and activations
        // the first z is Z 2
        for (int i = 0; i < z.length; i++) {
            z[i] = Matrix.dot(weight[i], a[i]).add(bias[i]);
            a[i + 1] = Matrix.c(z[i])
                    .apply(i == z.length - 1 ? outputActivationFunction : activationFunction, false);
        }

        return new Pair<>(z, a);
    }

    /**
     * easy to use feedForward method
     *
     * @param input input column vector
     * @return output column double array
     */
    public double[] feedForward(double[] input) {
        var outputs = feedForward(Matrix.columnVector(input));
        return ArrayUtils.lastElement(outputs.b).flatten();
    }


    /**
     * train the network:
     * 1. feed forward
     * 2. back propagate
     * 3. update weights
     * 4. repeat
     * <p>
     * this function loops through the epochs and shuffles the input and output data
     * (in the same manner so that the order is still correct),
     * takes a batch of data and trains the network on it
     * (calls fit(Matrix[], Matrix[]))
     *
     * @param inputs    column vectors of inputs
     * @param outputs   column vectors of outputs
     * @param epochs    number of epochs
     * @param batchSize size of batch for mini-batch gradient descent
     * @return loss history
     */
    public double[] fit(Matrix[] inputs, Matrix[] outputs, int batchSize, int epochs) {
        if (batchSize > inputs.length)
            throw new IllegalArgumentException("batch size cannot be greater than inputs length");

        double currentLoss;

        List<Double> loss = new ArrayList<>();
        for (int i = 0; i < epochs; i++) {
            // create mini batches by shuffling input
            Matrix[] batchInputs, batchOutputs;
            if (batchSize != inputs.length) {
                ArrayUtils.shuffle(inputs, outputs);

                batchInputs = new Matrix[batchSize];
                batchOutputs = new Matrix[batchSize];

                for (int j = 0; j < batchSize; j++) {
                    batchInputs[j] = inputs[j];
                    batchOutputs[j] = outputs[j];
                }
            } else {
                batchInputs = inputs;
                batchOutputs = outputs;
            }

            loss.add(currentLoss = fit(batchInputs, batchOutputs));

            // print the loss
            Log.l("Epoch " + i + ": " + currentLoss);
        }

        return ArrayUtils.toPrimitive(loss.toArray(new Double[0]));
    }

    /**
     * train one epoch
     *
     * @param X column vectors of inputs
     * @param Y column vectors of outputs
     * @return loss
     */
    public double fit(Matrix[] X, Matrix[] Y) {
        if (X.length != Y.length)
            throw new IllegalArgumentException("inputs and outputs must be of same length");

        double m = X.length;
        double L = 0;

        Matrix[] accumulatedWeightUpdates = new Matrix[weight.length],
                accumulatedBiasUpdates = new Matrix[bias.length],
                deltas = new Matrix[weight.length];

        // init deltas
        for (int i = 0; i < accumulatedWeightUpdates.length; i++) {
            accumulatedWeightUpdates[i] = Matrix.zeros(weight[i].rows, weight[i].cols);
            accumulatedBiasUpdates[i] = Matrix.zeros(bias[i].rows, bias[i].cols);
        }

        for (int i = 0; i < m; i++) {
            // feed forward
            var feedForward = feedForward(X[i]);
            Matrix[] z = feedForward.a, a = feedForward.b;

            // calculate last delta
            deltas[deltas.length - 1] = ArrayUtils.lastElement(a)
                    .subtract(Y[i]);

            // sum up loss
            L += ArrayUtils.lastElement(deltas).l2norm();


            // calculate other deltas
            for (int j = deltas.length - 2; j >= 0; j--) {
                deltas[j] = Matrix.dot(Matrix.transpose(weight[j + 1]), deltas[j + 1])
                        .multiply(z[j].apply(activationFunction, true));
            }

            // add to accumulated weight and bias updates
            // delta times activation for weights
            for (int j = 0; j < deltas.length; j++) {
                accumulatedWeightUpdates[j].add(
                        Matrix.dot(deltas[j], Matrix.transpose(a[j]))
                );

                accumulatedBiasUpdates[j].add(deltas[j]);
            }
        }

        // update weights and biases
        // multiply by 1 / m * learning rate
        for (int i = 0; i < weight.length; i++) {
            weight[i] = weight[i]
                    .subtract(accumulatedWeightUpdates[i].multiply(learningRate / m));
            bias[i] = bias[i]
                    .subtract(accumulatedBiasUpdates[i].multiply(biasLearningRate / m));
        }

        // return loss (average)
        return L / m;
    }

    /**
     * easy to use network training
     *
     * @param inputs    input column vectors
     * @param outputs   output column vectors
     * @param batchSize size of batch for mini-batch gradient descent
     * @param epochs    number of epochs
     * @return loss history
     */
    public double[] fit(double[][] inputs, double[][] outputs, int batchSize, int epochs) {
        Matrix[] input = new Matrix[inputs.length],
                output = new Matrix[outputs.length];

        for (int i = 0; i < inputs.length; i++) {
            input[i] = Matrix.columnVector(inputs[i]);
            output[i] = Matrix.columnVector(outputs[i]);
        }

        return fit(input, output, batchSize, epochs);
    }

    /**
     * print the structure of the neural network
     */
    public void printNetwork() {
        for (int i = 0; i < layerStructure.length - 1; i++) {
            System.out.println("Layer " + i);
            System.out.println("Bias vector: \n" + bias[i]);
            System.out.println("Weight matrix: \n" + weight[i]);
        }
    }
}
