package mlp;

import mlp.activationfunction.ActivationFunction;
import mlp.matrix.Matrix;

import java.io.Serializable;

// https://www.youtube.com/watch?v=x_Eamf8MHwU
public class MLP2 implements Serializable {
    /**
     * thetas, weight matrices
     * augmented weight vectors are used
     */
    public Matrix[] weights;

    /**
     * activation function for all layers
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
     * @param input column vector of y inputs
     * @return column vectors of outputs for each layer
     */
    public Matrix[] feedForward(Matrix input) {
        Matrix[] outputs = new Matrix[layerStructure.length];
        outputs[0] = input;

        for (int i = 0; i < outputs.length - 1; i++)
            outputs[i + 1] = Matrix.dot(weights[i], outputs[i])
                    .apply(activationFunction, false);

        return outputs;
    }

    /**
     * easy to use feedForward method
     * @param input input column vector
     * @return output column double array
     */
    public double[] feedForward(double[] input) {
        Matrix[] outputs = feedForward(Matrix.columnVector(input));
        return outputs[outputs.length - 1].flatten();
    }


    /**
     * train the network
     * @param inputs column vectors of inputs
     * @param outputs column vectors of outputs
     * @param epochs number of epochs
     * @param batchSize size of batch for mini-batch gradient descent
     * @return loss history
     */
    public double[] fit(Matrix[] inputs, Matrix[] outputs, int batchSize, int epochs) {

    }

    /**
     * easy to use network training
     * @param inputs input column vectors
     * @param outputs output column vectors
     * @param batchSize size of batch for mini-batch gradient descent
     * @param epochs number of epochs
     * @return loss history
     */
    public double[] fit(double[][] inputs, double[][] outputs, int batchSize, int epochs) {
        Matrix[] input = new Matrix[inputs.length],
                output = new Matrix[outputs.length];

        for(int i = 0; i < inputs.length; i++) {
            input[i] = Matrix.columnVector(inputs[i]);
            output[i] = Matrix.columnVector(outputs[i]);
        }

        return fit(input, output, batchSize, epochs);
    }
}
