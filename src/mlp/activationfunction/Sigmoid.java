package mlp.activationfunction;

public class Sigmoid implements ActivationFunction {

    @Override
    public double activate(double input) {
        return (1. / (1. + Math.pow(Math.E, -input)));
    }

    @Override
    public double derive(double input) {
        double sigmoid = activate(input);
        return sigmoid * (1 - sigmoid);
    }
}
