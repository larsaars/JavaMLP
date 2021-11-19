package test.patternrecognition;

import mlp.NeuralNetwork;
import mlp.utils.Log;
import mlp.utils.NNUtils;

import java.util.Arrays;

public class PatternRecognitionTestUse {
    public static void main(String[] args) {
        // load network
        NeuralNetwork nn = NNUtils.load();

        // load image and classify
        double[] image = PatternRecognitionTestTrain.loadImage("img/16.png", 28, 28, false);
        double[] output = nn.predict(image);

        Log.l(Arrays.toString(output));
    }
}
