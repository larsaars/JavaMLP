package test;

import mlp.MLP;
import mlp.MLP2;
import mlp.activationfunction.ActivationFunctions;
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
        MLP2 nn = new MLP2(new int[]{2, 16, 16, 16, 16, 16, 1}, ActivationFunctions.SIGMOID, 1e-4);

        double[] loss = nn.fit(X, Y, 1e-6, 10000);
        for (double[] d : new double[][]{
                {1, 1},
        }) {
            double[] output = nn.predict(d);
            System.out.println(Arrays.toString(output));
        }

        NNUtils.save(nn, loss);
    }
}