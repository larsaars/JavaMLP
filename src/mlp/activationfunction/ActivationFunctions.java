package mlp.activationfunction;

public class ActivationFunctions {
    public static final ActivationFunction SIGMOID = new Sigmoid(),
            HEAVISIDE = new Heaviside(),
            HYPERBOLIC_TANGENT = new HyperbolicTangent(),
            IDENTITY = new Identity(),
            RELU = new ReLU(),
            SOFTPLUS = new SoftPlus(),
            SOFTSIGN = new SoftSign(),
            LEAKY_RELU = new LeakyReLU();
}
