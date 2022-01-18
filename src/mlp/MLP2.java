package mlp;

import mlp.activationfunction.ActivationFunction;
import mlp.matrix.ArrayUtils;
import mlp.matrix.Matrix;
import mlp.utils.Pair;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

// https://www.youtube.com/watch?v=x_Eamf8MHwU
public class MLP2 implements Serializable {
    /**
     * thetas, weight matrices
     * augmented weight vectors are used
     */
    public Matrix[] weights;

    /**
     * activation function for all layers
     * g(x)
     */
    public ActivationFunction activationFunction;

    /**
     * layer structure
     */
    public int[] layerStructure;

    /**
     * hyperparameter learning rate eta
     */
    public double learningRate;

    /**
     * initializer
     *
     * @param layerStructure     layer structure
     * @param activationFunction activation function
     * @param learningRate       learning rate
     */
    public MLP2(int[] layerStructure, ActivationFunction activationFunction, double learningRate) {
        this.learningRate = learningRate;
        this.layerStructure = layerStructure;
        this.activationFunction = activationFunction;

        // initialize weights
        weights = new Matrix[layerStructure.length - 1];

        for (int i = 0; i < weights.length; i++)
            // one column more for bias
            weights[i] = Matrix.random(layerStructure[i + 1], layerStructure[i] + 1);
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
        for (int i = 0; i < z.length - 1; i++) {
            z[i] = Matrix.dot(weights[i], a[i]);
            a[i + 1] = Matrix.c(z[i]).apply(activationFunction, false);
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

        List<Double> loss = new ArrayList<>();
        for (int i = 0; i < epochs; i++) {
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

            loss.add(fit(batchInputs, batchOutputs));
        }

        return ArrayUtils.toPrimitive(loss.toArray(new Double[0]));
    }

    /**
     * train one epoch
     *
     * @param inputs  column vectors of inputs
     * @param outputs column vectors of outputs
     * @return loss
     */
    public double fit(Matrix[] inputs, Matrix[] outputs) {
        if (inputs.length != outputs.length)
            throw new IllegalArgumentException("inputs and outputs must be of same length");

        double m = inputs.length;
        double L = 0;

        Matrix[] accumulatedDeltas = new Matrix[weights.length],
                deltas = new Matrix[weights.length];

        // init deltas
        for (int i = 0; i < accumulatedDeltas.length; i++)
            accumulatedDeltas[i] = Matrix.zeros(weights[i].rows, weights[i].cols);

        for (int i = 0; i < m; i++) {
            // feed forward
            var feedForward = feedForward(inputs[i]);
            Matrix[] z = feedForward.a, a = feedForward.b;

            // calculate last delta
            deltas[deltas.length - 1] = ArrayUtils.lastElement(a)
                    .subtract(outputs[i]);

            // sum up loss
            L += ArrayUtils.lastElement(deltas).l2norm();

            // calculate other deltas
            for (int j = deltas.length - 2; j >= 0; j--) {
                deltas[j] = Matrix.dot(Matrix.transpose(weights[j]), deltas[j + 1])
                        .multiply(z[j - 1].apply(activationFunction, true));
            }

            // add to accumulated deltas
            for (int j = 0; j < deltas.length; j++)
                accumulatedDeltas[j].add(deltas[j]);
        }

        // update weights
        // multiply by 1 / m
        // and multiply all columns but the first one with learning rate
        // (the first one is the bias)
        for (int i = 0; i < weights.length; i++) {
            weights[i] = weights[i]
                    .subtract(accumulatedDeltas[i].multiply(1 / m).multiplyExceptColumn(learningRate, 0));
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
}
