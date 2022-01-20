package mlp.activationfunction;

public class Sigmoid implements ActivationFunction {

    @Override
    public double activate(double z) {
        return (1. / (1. + Math.exp(-z)));
    }

    @Override
    public double derive(double z) {
        double sigmoid = activate(z);
        return sigmoid * (1 - sigmoid);
    }
}
