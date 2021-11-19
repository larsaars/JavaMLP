package test.patternrecognition;

import mlp.NeuralNetwork;
import mlp.matrix.ArrayUtils;
import mlp.utils.Log;
import mlp.utils.NNUtils;

import java.util.Arrays;

/**
 * "testing" the pattern recognition system
 * please note that this is not a real test, it is just a way to test the system
 * and see if it works.
 * In a real test, you would have to split the data into training and test sets,
 * with a ratio of 80:20 for example.
 */
public class PatternRecognitionTestUse {

    private static final char[] PATTERNS =
            "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            .toCharArray();

    public static void main(String[] args) {
        // load network
        NeuralNetwork nn = NNUtils.load();

        // load image and classify
        double[] image = PatternRecognitionTestTrain.loadImage("img/16.png", 28, 28, false);
        double[] output = nn.predict(image);

        Log.l(Arrays.toString(output));
        Log.l(PATTERNS[ArrayUtils.argMax(output)]);
    }
}
