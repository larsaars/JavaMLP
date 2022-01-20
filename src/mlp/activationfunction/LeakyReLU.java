package mlp.activationfunction;

public class LeakyReLU implements ActivationFunction {

    private static final double ALPHA = 0.01;

    @Override
    public double activate(double z) {
        return Math.max(ALPHA * z, z);
    }

    @Override
    public double derive(double z) {
        return z > 0 ? 1 : ALPHA;
    }
}
