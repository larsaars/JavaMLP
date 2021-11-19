package mlp.utils;


public class Log {
    public static final boolean DEBUG = true;

    public static Object[] m(Object... os) {
        for (Object o : os)
            l(o);

        return os;
    }

    public static <T> T l(T o) {
        String s = String.valueOf(o);

        if (o == null)
            s = "null";

        if (DEBUG) {
            if (o instanceof Throwable) {
                System.err.println(getCallerClassName() + " -> error");
                ((Throwable) o).printStackTrace();
            } else {
                System.out.println(getCallerClassName() + " -> " + s);
            }
        }

        return o;
    }

    public static void l(String format, Object... os) {
        l(String.format(format, os));
    }

    private static String getCallerClassName() {
        StackTraceElement[] stElements = Thread.currentThread().getStackTrace();
        for (int i = 1; i < stElements.length; i++) {
            StackTraceElement ste = stElements[i];
            if (!ste.getClassName().equals(Log.class.getName()) && ste.getClassName().indexOf("java.lang.Thread") != 0) {
                return ste.getClassName() + "." + ste.getMethodName() + ": " + ste.getLineNumber();
            }
        }
        return "UNKNOWN";
    }
}