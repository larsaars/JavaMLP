package mlp.matrix;

public class ShapeMismatchException extends RuntimeException{
    public ShapeMismatchException(String format, Object... args) {
        super(String.format(format, args));
    }
}
