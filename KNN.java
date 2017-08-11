import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.Text;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

/**
 * Created by yijia on 2/14/17.
 */

public class KNN {
    public static class KNNMapper extends Mapper<Object, Text, Text, Text> {

        private ArrayList<ArrayList> test = new ArrayList<ArrayList>();

        @Override
        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {

            System.out.println("map begin");
            String[] inarr = value.toString().split(",");
            String l = inarr[inarr.length - 1];
            for (int i = 0; i < test.size(); i++) {
                double dist = Edist(test.get(i), value);
                String valueout = inarr[inarr.length - 1] + "," + Double.toString(dist); // label + dist
                context.write(new Text(i + ""), new Text(valueout));
                System.out.println("i = " + i + " value out :" + valueout);
            }
            System.out.println("map end");
        } // map

        public double Edist(ArrayList<Double> t, Text v) {
            System.out.println("AL t: " + Arrays.toString(t.toArray()));
            String[] value = v.toString().split(",");
            System.out.println("Text :" + Arrays.toString(value));
            int res = 0;
            for (int i = 0; i < value.length - 1; i++) {
                res += Math.pow(Double.parseDouble(value[i])-t.get(i), 2);
            }
            System.out.println("edist: " + Math.sqrt(res));
            return Math.sqrt(res);
        } // Edist

        protected void setup(Context context) throws java.io.IOException, InterruptedException {

            //FileSystem fs = FileSystem.get(context.getConfiguration());
            BufferedReader br = new BufferedReader(new FileReader("/Users/yijia/Downloads/iris_test_data.csv"));
            String line = br.readLine();
            while (line != null) {
                String[] ss = line.split(",");
                ArrayList<Double> arr = new ArrayList<Double>();
                for (String s : ss) {
                    arr.add(Double.parseDouble(s));
                }
                test.add(arr);
                line = br.readLine();
            }
            br.close();
            System.out.println("map setup");
            for (int i = 0; i < test.size(); i++) {
                for (int j = 0; j < test.get(i).size(); j++) {
                    System.out.print(test.get(i).get(j)+" ");
                }
                System.out.println();
            }
        } // readTest
    } // Mapper


    public static class KNNReducer
            extends Reducer<Text, Text, Text, Text> {

        public class Node {
            String label;
            double dist;

            public Node(String l, double d) {
                this.label = l;
                this.dist = d;
            }
        }

        public void reduce(Text key, Iterable<Text> value, Context context)
                throws IOException, InterruptedException {

            System.out.println("reduce begin: key = " + key.toString());

            ArrayList<Node> nodes = new ArrayList<Node>();
            //ArrayList<Double> dist = new ArrayList<Double>();
            for (Text v : value) {
                System.out.println("value = " + v.toString());
                String str = v.toString();
                String[] s = str.split(",");
                nodes.add(new Node(s[0], Double.parseDouble(s[1])));
            } // for

            int k = 5;
            Collections.sort(nodes, new Comparator<Node>() {
                public int compare(Node n1, Node n2) {
                    //System.out.println("compare begin");
                    //System.out.println("nodes:" + n1.dist + n2.dist);
                    if (n1.dist < n2.dist) return -1;
                    else if (n1.dist > n2.dist) return 1;
                    else return 0;
                }
            });
            System.out.println("sorted:");
            nodes.forEach(i -> System.out.println("node: label:" + i.label + "\tdist = " + i.dist));

            int[] cnt = new int[5];
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    if (nodes.get(i).label == nodes.get(j).label) cnt[j]++;
                }
            }
            int max = -1, idx = 0;
            String pre = "";
            for (int i = 0; i < 5; i++) {
                if (cnt[i] > max) {
                    max = cnt[i];
                    idx = i;
                    pre = nodes.get(i).label;
                    System.out.println("pre:" + pre);
                }
            }

            context.write(key, new Text(pre));
            System.out.println("write : " + pre);
            System.out.println("reduce end");
        } // reduce
    } // Reducer

    public static void main(String[] args) throws Exception {

        System.out.println("main begin");
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "KNN");
        job.setJarByClass(KNN.class);
        job.setMapperClass(KNNMapper.class);
        job.setReducerClass(KNNReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
        System.out.println("main end");
    }

}
