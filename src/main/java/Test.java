import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.opencsv.CSVWriter;
import com.yahoo.labs.samoa.instances.*;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.functions.AdaGrad;
import moa.classifiers.functions.SGD;
import moa.classifiers.functions.SGDMultiClass;
import moa.classifiers.lazy.kNN;
import moa.classifiers.lazy.kNNwithPAW;
import moa.classifiers.lazy.kNNwithPAWandADWIN;
import moa.classifiers.meta.AdaptiveRandomForestRegressor;
import moa.classifiers.meta.RandomRules;
import moa.classifiers.rules.AMRulesRegressor;
import moa.classifiers.rules.functions.*;
import moa.classifiers.rules.meta.RandomAMRules;
import moa.classifiers.trees.*;
import moa.core.Example;
import moa.core.InstanceExample;
import moa.core.TimingUtils;
import moa.evaluation.BasicRegressionPerformanceEvaluator;
import moa.learners.featureanalysis.FeatureImportanceHoeffdingTree;
import moa.options.ClassOption;
import moa.streams.ArffFileStream;
import mst.In;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Stack;
import java.util.concurrent.ThreadLocalRandom;

public class Test {

    public static final ArffFileStream stream = new ArffFileStream("./data/multivariate_2/wind_data.arff", 406);
    //public static MovingWindowStream stream = new MovingWindowStream("./data/no_window_good/data_no_timestamp.txt", 1, 1);
    public static final int LIMIT = 5;

    public static <T extends AbstractClassifier> void run(T learner, String name) {
        try {
            CSVWriter writer = new CSVWriter(new FileWriter("./data/multivariate_2/output_wind_" + name + ".csv"));
            CSVWriter metrics = new CSVWriter(new FileWriter("./data/multivariate_2/metrics_wind.csv", true));
            FeatureImportanceHoeffdingTree featureSelector = new FeatureImportanceHoeffdingTree();

            stream.restart();
            BasicRegressionPerformanceEvaluator evaluator = new BasicRegressionPerformanceEvaluator();

            Instances insts = new Instances("train", computeAttributes(), LIMIT);
            InstancesHeader header = new InstancesHeader(insts);

            featureSelector.setModelContext(stream.getHeader());
            featureSelector.prepareForUse();
            learner.setModelContext(header);
            learner.prepareForUse();
            int numberSamples = 0;
            long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
            Example<Instance> currentInst;
            Example<Instance> trainInst;
            int[] topFeatures = new int[LIMIT];

            while(stream.hasMoreInstances()) {
                currentInst = stream.nextInstance();
                numberSamples++;

                if(numberSamples == 1) {
                    featureSelector.trainOnInstance(currentInst);
                    topFeatures = featureSelector.getTopKFeatures(LIMIT, false);
                }

                trainInst = prepareInstance(currentInst, topFeatures, header);

                Prediction pred = learner.getPredictionForInstance(trainInst);

                if(numberSamples != 1) {
                    featureSelector.trainOnInstance(currentInst);
                    topFeatures = featureSelector.getTopKFeatures(LIMIT, false);
                }

                trainInst = prepareInstance(currentInst, topFeatures, header);

                learner.trainOnInstance(trainInst);

                if(numberSamples > 1) {
                    evaluator.addResult(trainInst, pred);
                    writer.writeNext(new String[]{String.valueOf(pred.getVotes()[0]), String.valueOf(evaluator.getMeanError()), String.valueOf(evaluator.getSquareError())}, false);
                }
                System.out.println(Arrays.toString(topFeatures));
                System.out.println(numberSamples);
            }

            writer.close();

            double time = TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread() - evaluateStartTime);
            System.out.println(numberSamples + " instances in "+time+" seconds with " + learner.getClass().getSimpleName() + ".");
            metrics.writeNext(new String[]{name, String.valueOf(time), String.valueOf(evaluator.getMeanError()), String.valueOf(evaluator.getSquareError())}, false);
            metrics.close();
            System.out.println(evaluator.getSquareError());
            System.out.println(evaluator.getMeanError());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static List<Attribute> computeAttributes() {
        List<Attribute> attributes = new ArrayList<>();

        for(int i = LIMIT - 1; i >= 0; i--) {
            attributes.add(new Attribute(i != 0 ? "t-" + i : "t"));
        }

        attributes.add(new Attribute("t+" + 1));

        return attributes;
    }

    public static InstanceExample prepareInstance(Example<Instance> inst, int[] topFeatures, InstancesHeader header) {
        Instance instRes;
        Stack<Double> data = new Stack<>();
        double[] dataArray;
        InstanceExample res;

        for(int feature : topFeatures) {
            double value = inst.getData().value(feature);
            data.push(value);
        }

        dataArray = data.stream().mapToDouble(Double::doubleValue).toArray();
        instRes = new InstanceImpl(1.0, dataArray);
        instRes.setDataset(header);
        res = new InstanceExample(instRes);

        return res;
    }

    public static void main(String[] args) {
        run(new AdaGrad(), "AdaGrad");
//        run(new AdaptiveNodePredictor(), "AdaptiveNodePredictor");
//        run(new AdaptiveRandomForestRegressor(), "AdaptiveRandomForestRegressor");
//        run(new AMRulesRegressor(), "AMRulesRegressor");
//        run(new ARFFIMTDD(), "ARFFIMTDD");
//        run(new FadingTargetMean(), "FadingTargetMean");
//        run(new FIMTDD(), "FIMTDD");
//        run(new kNN(), "kNN");
//        run(new TargetMean(), "TargetMean");
//        run(new kNNwithPAW(), "kNNwithPAW");
//        run(new kNNwithPAWandADWIN(), "kNNwithPAWandADWIN");
//        run(new LowPassFilteredLearner(), "LowPassFilteredLearner");
//        run(new ORTO(), "ORTO");
//        run(new Perceptron(), "Perceptron");
//        run(new RandomAMRules(), "RandomAMRules");
//        run(new RandomRules(), "RandomRules");
//        run(new SGD(), "SGD");
//        run(new SGDMultiClass(), "SGDMultiClass");

//        for(float eps = 1e-8f; eps < 3f; eps += 1f) {
//            for(float lambdaReg = 0.0001f; lambdaReg <= 0.01f; lambdaReg += 0.0025f) {
//                for(float learningRate = 0.0001f; learningRate < 0.02f; learningRate += 0.005f) {
//                    for(int i = 0; i < 3; i++) {
//                        AdaGrad adaGrad = new AdaGrad();
//                        adaGrad.epsilonOption = new FloatOption("epsilon", 'p', "epsilon parameter.", eps);
//                        adaGrad.lambdaRegularizationOption = new FloatOption("lambdaRegularization", 'l', "Lambda regularization parameter .", lambdaReg, 0.00, Integer.MAX_VALUE);
//                        adaGrad.learningRateOption = new FloatOption("learningRate",'r', "Learning rate parameter.",learningRate, 0.00, Integer.MAX_VALUE);
//                        adaGrad.lossFunctionOption = new MultiChoiceOption("lossFunction", 'o', "The loss function to use.", new String[]{"HINGE", "LOGLOSS", "SQUAREDLOSS"}, new String[]{"Hinge loss (SVM)", "Log loss (logistic regression)", "Squared loss (regression)"}, i);
//                        run(adaGrad, "AdaGrad_"+eps+"_"+lambdaReg+"_"+learningRate+"_"+i);
//                    }
//                }
//            }
//        }
//
//        run(new AdaptiveNodePredictor(), "AdaptiveNodePredictor");
//
//        for(float alpha = 0.15f; alpha < 1; alpha += 0.15f) {
//            LowPassFilteredLearner lp = new LowPassFilteredLearner();
//            lp.baseLearnerOption = new ClassOption("baseLearner", 'l',"Base learner.", AbstractClassifier.class, AdaGrad.class.getName());
//            lp.alphaOption = new FloatOption("alpha", 'a',  "Alpha value. Y=Yold+alpha*(Yold+Prediction)", alpha, 0, 1);
//            run(lp, "LowPassFilteredLearner_"+alpha);
//        }
//
//        for(int tf = 0; tf < 2; tf++) {
//            for(float lr = 0.025f; lr < 0.1f; lr += 0.025f) {
//                for(float lrd = 0.001f; lrd < 0.01f; lrd += 0.003f) {
//                    for(float ff = 0.001f; ff < 1f; ff += 0.25f) {
//                        for(int i = 0; i < 4; i++) {
//                            Perceptron p = new Perceptron();
//                            if(tf == 0) {
//                                p.constantLearningRatioDecayOption.unset();
//                            } else {
//                                p.constantLearningRatioDecayOption.set();
//                            }
//                            p.learningRatioOption = new FloatOption("learningRatio", 'l',"Constante Learning Ratio to use for training the Perceptrons in the leaves.", lr);
//                            p.learningRateDecayOption = new FloatOption("learningRateDecay", 'm'," Learning Rate decay to use for training the Perceptron.", lrd);
//                            p.fadingFactorOption = new FloatOption("fadingFactor", 'e', "Fading factor for the Perceptron accumulated error", ff, 0, 1);
//                            int random = ThreadLocalRandom.current().nextInt(1, 10000 + 1);
//                            p.randomSeedOption = new IntOption("randomSeed", 'r', "Seed for random behaviour of the classifier.", random);
//                            run(p, "Perceptron_"+tf+"_"+lr+"_"+lrd+"_"+ff+"_"+random);
//                        }
//                    }
//                }
//            }
//        }
//
//        for(int nn = 10; nn < 25; nn += 5) {
//            for(int tf = 0; tf < 2; tf++) {
//                //for(int limit = 400; limit <= 1000; limit += 200) {
//                    //for(int option = 0; option < 2; option++) {
//                        kNN knn = new kNN();
//                        knn.kOption = new IntOption( "k", 'k', "The number of neighbors", nn, 1, Integer.MAX_VALUE);
//                        if(tf == 0) {
//                            knn.medianOption.unset();
//                        } else {
//                            knn.medianOption.set();
//                        }
//                        //knn.limitOption = new IntOption( "limit", 'w', "The maximum number of instances to store", limit, 1, Integer.MAX_VALUE);
//                        //knn.nearestNeighbourSearchOption = new MultiChoiceOption("nearestNeighbourSearch", 'n', "Nearest Neighbour Search to use", new String[]{"LinearNN", "KDTree"},new String[]{"Brute force search algorithm for nearest neighbour search. ", "KDTree search algorithm for nearest neighbour search"}, option);
//                        run(knn, "kNN_"+nn+"_"+tf);
//                    //}
//                //}
//            }
//        }
//
//        for(int nn = 10; nn < 25; nn += 5) {
//            for(int tf = 0; tf < 2; tf++) {
//                //for(int limit = 400; limit <= 1000; limit += 200) {
//                    //for(int option = 0; option < 2; option++) {
//                        kNNwithPAW knnPaw = new kNNwithPAW();
//                        knnPaw.kOption = new IntOption( "k", 'k', "The number of neighbors", nn, 1, Integer.MAX_VALUE);
//                        if(tf == 0) {
//                            knnPaw.medianOption.unset();
//                        } else {
//                            knnPaw.medianOption.set();
//                        }
//                        //knnPaw.limitOption = new IntOption( "limit", 'w', "The maximum number of instances to store", limit, 1, Integer.MAX_VALUE);
//                        //knnPaw.nearestNeighbourSearchOption = new MultiChoiceOption("nearestNeighbourSearch", 'n', "Nearest Neighbour Search to use", new String[]{"LinearNN", "KDTree"},new String[]{"Brute force search algorithm for nearest neighbour search. ", "KDTree search algorithm for nearest neighbour search"}, option);
//                        run(knnPaw, "kNNwithPAW_"+nn+"_"+tf);
//                    //}
//                //}
//            }
//        }

    }

}
