package test.patternrecognition;

import mlp.MLP2;
import mlp.activationfunction.ActivationFunctions;
import mlp.matrix.ArrayUtils;
import mlp.utils.NNUtils;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static mlp.utils.Log.*;


public class PatternRecognitionTestTrain {

    public static void main(String[] args) {
        // load the training data
        List<Double[]> X_list = new ArrayList<>(),
                Y_list = new ArrayList<>();

        // folders with patterns and samples
        for (int i = 1; i <= 62; i++) {
            for (int j = 1; j <= 88; j++) {
                double[] image = loadImage("img/patterns/" + i + "/" + j + ".png", 28, 28, false);

                Double[] y = ArrayUtils.toObject(ArrayUtils.oneHotEncoding(62, i - 1));
                Double[] x = ArrayUtils.toObject(image);

                X_list.add(x);
                Y_list.add(y);
            }

            l("Loaded " + i + " pattern folders");
        }

        double [][] X = ArrayUtils.fromList(X_list),
                Y = ArrayUtils.fromList(Y_list);

        // create the nn instance
       MLP2 nn = new MLP2(new int[]{784, 70, 70, 70, 62}, ActivationFunctions.SIGMOID, ActivationFunctions.IDENTITY, 1e-3, 0.5);
        // fit data and save loss and model
        double[] loss = nn.fit(
                X,
                Y,
                100,
               600
        );

        l("saving nn and loss");
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