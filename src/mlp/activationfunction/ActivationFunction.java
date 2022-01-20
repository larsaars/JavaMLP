package mlp.activationfunction;

import java.io.Serializable;

public interface ActivationFunction extends Serializable {
    double activate(double z);

    double derive(double z);
}