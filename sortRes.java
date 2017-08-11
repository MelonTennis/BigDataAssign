/**
 * Created by yijia on 2/16/17.
 */

import java.lang.*;
import java.util.*;
import java.io.*;

public class sortRes {

    public static void main(String[] args) throws IOException {
        String[] res = new String[30];
        BufferedReader br = new BufferedReader(new FileReader("/Users/yijia/IdeaProjects/KNNr/output/part-r-00000"));
        String line = br.readLine();
        while (line != null) {
            String[] ss = line.split("\\s+");
            int idx = Integer.parseInt(ss[0]);
            res[idx] = ss[1];
            line = br.readLine();
            System.out.println(res[idx]);
        }
        br.close();

        BufferedWriter out=new BufferedWriter(new FileWriter("/Users/yijia/IdeaProjects/KNNr/output/sortedRes"));
        for(String s: res){
            out.write(s);
            out.newLine();
        }
        out.close();
    }

}
