package mlp.activationfunction;

public class HyperbolicTangent implements ActivationFunction {

    @Override
    public double activate(double input) {
        double epx = Math.pow(Math.E, input);
        double enx = Math.pow(Math.E, -input);

        return ((epx - enx) / (epx + enx));
    }

    @Override
    public double derive(double input) {
        double tanh = activate(input);

        return 1 - Math.pow(tanh, 2);
    }
}
