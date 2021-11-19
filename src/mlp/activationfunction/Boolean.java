package mlp.activationfunction;

public class Boolean implements ActivationFunction {

    @Override
    public double activate(double input) {
        if (input < 0) return 0;
        else return 1;
    }

    @Override
    public double derive(double input) {
        return 1;
    }
}
