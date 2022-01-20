package test;

import mlp.utils.NNUtils;
import mlp.MLP;
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
        MLP nn = new MLP(new int[]{2, 16, 16, 16, 16, 16, 1}, 1e-3, ActivationFunctions.LEAKY_RELU);

        double[] loss = nn.fit(X, Y, 100000);
        for (var d : X) {
            double[] output = nn.predict(d);
            System.out.println(Arrays.toString(output));
        }

        NNUtils.save(nn, loss);
    }
}