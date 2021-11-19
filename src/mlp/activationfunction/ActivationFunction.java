package mlp.activationfunction;

import java.io.Serializable;

public interface ActivationFunction extends Serializable {
    double activate(double input);

    double derive(double input);
}