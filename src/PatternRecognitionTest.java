import mlp.NeuralNetwork;
import mlp.activationfunction.ActivationFunctions;
import mlp.matrix.ArrayUtils;
import mlp.utils.NNUtils;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class PatternRecognitionTest {

    public static void main(String[] args) {
        // load the training data
        List<Double[]> X = new ArrayList<>(),
                Y = new ArrayList<>();

        // folders with patterns and samples
        for (int i = 1; i <= 62; i++) {
            for (int j = 1; j <= 88; j++) {
                double[] image = loadImage("img/patterns/" + i + "/" + j + ".png", 28, 28, false);

                Double[] y = ArrayUtils.toObject(ArrayUtils.oneHotEncoding(62, i - 1));
                Double[] x = ArrayUtils.toObject(image);

                X.add(x);
                Y.add(y);
            }
        }

        // create the nn instance
        NeuralNetwork nn = new NeuralNetwork(new int[]{784, 16, 16, 16, 62}, 1e-3, ActivationFunctions.RELU);
        // fit data and save loss and model
        double[] loss = nn.fit(
                ArrayUtils.fromList(X),
                ArrayUtils.fromList(Y),
                0.01,
                20,
                1000000
        );

        NNUtils.save(nn, loss);
    }


    public static double[] loadImage(String path, int sizeX, int sizeY, boolean color) {
        BufferedImage imgCopy, img;

        if (color)
            imgCopy = new BufferedImage(
                    sizeX,
                    sizeY,
                    BufferedImage.TYPE_INT_ARGB);
        else
            imgCopy = new BufferedImage(
                    sizeX,
                    sizeY,
                    BufferedImage.TYPE_BYTE_GRAY);

        try {
            img = ImageIO.read(new File(path));
        } catch (IOException ex) {
            System.out.println(path + " not loaded");
            return null;
        }


        Graphics2D g = imgCopy.createGraphics();
        g.drawImage(img, 0, 0, null);
        g.dispose();

        double[] data = new double[sizeX * sizeY];

        for (int i = 0; i < sizeX; i++)
            for (int j = 0; j < sizeY; j++) {
                int[] d = new int[3];
                imgCopy.getRaster().getPixel(i, j, d);

                data[i * sizeX + j] = ((double) d[0]) / 255.0;
            }

        return data;
    }
}