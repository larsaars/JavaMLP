package test.patternrecognition;

import mlp.MLP;
import mlp.matrix.ArrayUtils;
import mlp.utils.Log;
import mlp.utils.NNUtils;

import java.io.File;
import java.util.Objects;

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
        MLP nn = NNUtils.load();

        nn.printNetwork();

        // load image and classify
        for (File testFile : Objects.requireNonNull(new File("img/test").listFiles())) {
            double[] image = PatternRecognitionTestTrain.loadImage(testFile.getPath(), 28, 28, false);
            double[] output = nn.predict(image);

            Log.l(testFile.getName() + ": " + PATTERNS[ArrayUtils.argMax(output)]);
        }
    }
}
