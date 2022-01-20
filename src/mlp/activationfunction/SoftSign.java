package mlp.activationfunction;

public class SoftSign implements ActivationFunction {
    @Override
    public double activate(double z) {
        return z / (1. + Math.abs(z));
    }

    @Override
    public double derive(double z) {
        return 1. / Math.pow(1. + Math.abs(z), 2);
    }
}
