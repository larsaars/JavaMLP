package mlp.activationfunction;

public class SoftPlus implements ActivationFunction {

    @Override
    public double activate(double z) {
        return Math.log(1 + Math.exp(z));
    }

    @Override
    public double derive(double z) {
        return 1. / (1. + Math.exp(-z));
    }
}
