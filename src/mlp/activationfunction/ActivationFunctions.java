package mlp.activationfunction;

public class ActivationFunctions {
    public static final ActivationFunction SIGMOID = new Sigmoid(),
            BOOLEAN = new Boolean(),
            HYPERBOLIC_TANGENT = new HyperbolicTangent(),
            IDENTITY = new Identity(),
            RELU = new ReLU();
}
