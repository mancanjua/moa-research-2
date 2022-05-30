import com.yahoo.labs.samoa.instances.Instance;
import meka.experiment.evaluators.Evaluator;
import moa.classifiers.Classifier;
import moa.classifiers.bayes.NaiveBayes;
import moa.classifiers.core.splitcriteria.InfoGainSplitCriterion;
import moa.classifiers.core.splitcriteria.VarianceReductionSplitCriterion;
import moa.classifiers.functions.AdaGrad;
import moa.classifiers.functions.MajorityClass;
import moa.classifiers.functions.Perceptron;
import moa.classifiers.functions.SGDMultiClass;
import moa.classifiers.meta.AdaptiveRandomForest;
import moa.classifiers.meta.AdaptiveRandomForestRegressor;
import moa.classifiers.rules.functions.AdaptiveNodePredictor;
import moa.classifiers.trees.DecisionStump;
import moa.classifiers.trees.FIMTDD;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.Example;
import moa.core.TimingUtils;
import moa.evaluation.WindowRegressionPerformanceEvaluator;
import moa.options.ClassOption;
import moa.streams.ArffFileStream;
import moa.streams.CachedInstancesStream;
import moa.streams.generators.RandomRBFGenerator;
import moa.tasks.CacheShuffledStream;
import moa.tasks.NullMonitor;
import mst.In;
import org.kramerlab.autoencoder.neuralnet.rbm.BernoulliUnitLayer;

import java.lang.management.MonitorInfo;
import java.util.Arrays;

public class Test {

//    public static final ArffFileStream stream = new ArffFileStream("./data/census.arff", 64);
    public static MovingWindowStream stream = new MovingWindowStream("./data/data_test_2.txt", 1000, 1);

    public static void run() {
        AdaGrad learner = new AdaGrad();
        WindowRegressionPerformanceEvaluator evaluator = new WindowRegressionPerformanceEvaluator();

        learner.setModelContext(stream.getHeader());
        learner.prepareForUse();

//        evaluator.widthOption.setValue(12);

        int numberSamplesCorrect = 0;
        int numberSamples = 0;
        long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
        Example<Instance> trainInst;

        while(stream.hasMoreInstances()) {
            trainInst = stream.nextInstance();
            numberSamples++;

            learner.trainOnInstance(trainInst);
            evaluator.addResult(trainInst, learner.getPredictionForInstance(trainInst));
        }

        double time = TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread() - evaluateStartTime);
        System.out.println(numberSamples + " instances in "+time+" seconds.");
        System.out.println(Arrays.toString(evaluator.getPerformanceMeasurements()));
    }

    public static void main(String[] args) {
        run();
    }

}
