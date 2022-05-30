import com.yahoo.labs.samoa.instances.*;
import moa.AbstractMOAObject;
import moa.core.Example;
import moa.core.InstanceExample;
import moa.core.TimingUtils;
import moa.streams.InstanceStream;
import org.jfree.util.ArrayUtilities;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class MovingWindowStream extends AbstractMOAObject implements InstanceStream {

    private static final long serialVersionUID = 1L;

    private final int past;
    private final int future;

    private int streamPos;
    private Instances stream;

    public MovingWindowStream(String filePath, int past, int future) {
        this.past = past;
        this.future = future;
        this.streamPos = 0;
        prepareInstances(filePath);
    }

    private void prepareInstances(String filePath) {
        long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
        List<Double> data = readFile(filePath);
        Instances res = new Instances("timeseries", this.computeAttributes(), data.size() - this.past - this.future);
        res.setClassIndex(this.past);

//        IntStream.range(this.past - 1, data.size() - this.future)
//                 .boxed()
//                 .map(i -> new InstanceImpl(1.0, IntStream.rangeClosed(i - this.past + 1, i + this.future)
//                                                                 .boxed()
//                                                                 .parallel()
//                                                                 .map(data::get)
//                                                                 .mapToDouble(Double::doubleValue)
//                                                                 .toArray()))
//                 .forEach(inst -> {
//                     inst.setDataset(new InstancesHeader(res));
//                     res.add(inst);
//        });

        IntStream.range(this.past - 1, data.size() - this.future)
                .boxed()
                .map(i -> new InstanceImpl(1.0, data.subList(i - this.past + 1, i + this.future + 1).stream()
                        .mapToDouble(Double::doubleValue)
                        .toArray()))
                .forEach(inst -> {
                    inst.setDataset(new InstancesHeader(res));
                    res.add(inst);
                });

        this.stream = res;
        System.out.println(TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread() - evaluateStartTime));
    }

    private List<Attribute> computeAttributes() {
        long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
        List<Attribute> attributes = new ArrayList<>();

        for(int i = this.past - 1; i >= 0; i--) {
            attributes.add(new Attribute(i != 0 ? "t-" + i : "t"));
        }

        for(int i = 1; i <= this.future; i++) {
            attributes.add(new Attribute("t+" + i));
        }

        System.out.println(TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread() - evaluateStartTime));
        return attributes;
    }

    private List<Double> readFile(String filePath) {
        long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
        List<Double> res = new LinkedList<>();
        try {
            String line;
            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(filePath)));
            while((line = br.readLine()) != null) {
                res.add(Double.parseDouble(line));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        System.out.println(TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread() - evaluateStartTime));
        return res;
    }

    @Override
    public InstancesHeader getHeader() {
        return new InstancesHeader(this.stream);
    }

    @Override
    public long estimatedRemainingInstances() {
       return this.stream.numInstances() - this.streamPos;
    }

    @Override
    public boolean hasMoreInstances() {
        return this.streamPos < this.stream.numInstances();
    }

    @Override
    public Example<Instance> nextInstance() {
        return this.hasMoreInstances() ? new InstanceExample(this.stream.instance(this.streamPos++)) : null;
    }

    @Override
    public boolean isRestartable() {
        return true;
    }

    @Override
    public void restart() {
        this.streamPos = 0;
    }

    @Override
    public void getDescription(StringBuilder sb, int indent) {

    }
}
