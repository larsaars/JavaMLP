package mlp.utils;

import mlp.MLP;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

public class NNUtils {
    private static final String DEFAULT_LOSS_FILE_NAME = "loss.txt",
            DEFAULT_NN_FILE_NAME = "nn.ser";

    public static void save(MLP nn, double[] loss) {
        Serializer.serialize(nn, DEFAULT_NN_FILE_NAME);
        writeLossToFile(loss, DEFAULT_LOSS_FILE_NAME);
    }
    
    public static MLP load() {
        return MLP.load(DEFAULT_NN_FILE_NAME);
    }
    
    public static void writeLossToFile(double[] loss, String fileName) {
        File file = new File(fileName);
        StringBuilder sb = new StringBuilder();
        for (double d : loss) {
            sb.append(d).append(" ");
        }

        try {
            FileOutputStream fos = new FileOutputStream(file);
            fos.write(sb.toString().getBytes(StandardCharsets.UTF_8));
            fos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
