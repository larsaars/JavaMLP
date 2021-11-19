package test;

import mlp.utils.NNUtils;
import mlp.NeuralNetwork;
import mlp.activationfunction.ActivationFunctions;

import java.util.Arrays;

public class XOrTest {

    static double[][] X = {
            {0, 0},
            {1, 0},
            {0, 1},
            {1, 1}
    };
    static double[][] Y = {
            {0}, {1}, {1}, {0}
    };

    public static void main(String[] args) {
        NeuralNetwork nn = new NeuralNetwork(new int[]{2, 16, 16, 16, 16, 16, 1}, 1e-3, ActivationFunctions.HYPERBOLIC_TANGENT);

        double[] loss = nn.fit(X, Y, 1e-2, 20, 1000000);
        for (double[] d : X) {
            double[] output = nn.predict(d);
            System.out.println(Arrays.toString(output));
        }

        NNUtils.save(nn, loss);
    }
}