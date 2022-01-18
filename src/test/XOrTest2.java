package test;

import mlp.MLP;
import mlp.MLP2;
import mlp.activationfunction.ActivationFunctions;
import mlp.matrix.ArrayUtils;
import mlp.utils.Log;
import mlp.utils.NNUtils;

import java.util.Arrays;

public class XOrTest2 {

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
        MLP2 nn = new MLP2(new int[]{2, 16, 16, 16, 16, 16, 1}, ActivationFunctions.SIGMOID, 1e-3);

        double[] loss = nn.fit(X, Y, 4, 100);
        for (double[] d : X) {
            double[] output = nn.feedForward(d);
            System.out.println("Output: " + Arrays.toString(output));
        }

        Log.l("Loss: " + Arrays.toString(loss));

        // NNUtils.save(nn, loss);
    }
}