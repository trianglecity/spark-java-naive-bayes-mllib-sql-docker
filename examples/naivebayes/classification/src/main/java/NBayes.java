

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.api.java.function.MapFunction;

import scala.Tuple2;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import org.apache.spark.sql.SQLContext;
import org.apache.spark.ml.feature.StopWordsRemover;

import org.apache.spark.sql.functions;
import org.apache.spark.sql.Column;
import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.ml.linalg.SparseVector;
import org.apache.spark.ml.linalg.DenseVector;

import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.Vector;

import org.apache.spark.rdd.RDD;

public class NBayes {


	public  static void main(String[] args) throws Exception {
		
		SparkConf conf = new SparkConf().setAppName("NaiveBayes").setMaster("local");
		JavaSparkContext sc = new JavaSparkContext(conf);
				
		System.out.println("args[0]= " + args[0]);
		JavaRDD<String> lines  = sc.textFile(args[0],1);
		
		//JavaRDD<Tuple2<String,String>> labelWords = lines.map(
		//	new Function <String, Tuple2<String,String>>()  { 
              	//		@Override
              	//		public Tuple2<String,String> call(String s) throws Exception {
		//			String[] ss = s.split("\\t");
		//			
		//			return new Tuple2<String,String>(ss[0], ss[1]);
              	//		}
            	//	}
		//);
		
		// label -- message pairs
		JavaPairRDD<String, String> keyvalue = lines.mapToPair(new PairFunction<String, String, String>() {
            		@Override
            		public Tuple2<String, String> call(String s) {

				String[] ss = s.split("\\t");
					
				return new Tuple2<String, String>(ss[0], ss[1]);	
                		//return new Tuple2<String, Integer>(readName, Integer.valueOf(1));
            		}
        	});

		for (Tuple2<String, String> test : keyvalue.take(10)) 
           	{
               		System.out.println(test._1 + " <-> " +test._2);
               		
          	}

		// sql ROW
		JavaRDD<Row> sqlRows = lines.map(
			new Function <String, Row>()  { 
              			@Override
              			public Row call(String s) throws Exception {
					String[] ss = s.split("\\t");

					double label = -1;

					if ( ss[0].equals("spam") ){
						label = 0;
					} 
					if ( ss[0].equals("ham") ){
						label = 1;
					}
								
					
					return RowFactory.create(label, ss[1]);
              			}
            		}
		);
			
		StructType schema = new StructType(new StructField[]{
  				new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
  				new StructField("sentence", DataTypes.StringType, false, Metadata.empty())
		});

		SQLContext sqlCtx = new SQLContext(sc);
		Dataset<Row> linesDF = sqlCtx.createDataFrame(sqlRows, schema);

		
		Tokenizer tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words");
                Dataset<Row> wordsData = tokenizer.transform(linesDF);

		StopWordsRemover remover = new StopWordsRemover()
  						.setInputCol("words")
  						.setOutputCol("filteredwords");
		Dataset<Row> filteredWordData = remover.transform(wordsData);
		filteredWordData.show(true);
		
			

		int numFeatures = 100;
		//  HashingTF is a Transformer which takes  a bag of words and converts those sets into fixed-length feature vectors.
		HashingTF myHashingTF = new HashingTF()
  						.setInputCol("filteredwords")
  						.setOutputCol("rawFeatures")
  						.setNumFeatures(numFeatures);

		Dataset<Row> featurizedData = myHashingTF.transform(filteredWordData);
		featurizedData.show(true);
		
		Dataset<Row> labelvalue = featurizedData.select(featurizedData.col("label"), featurizedData.col("rawFeatures"));
		labelvalue.show(true);	

		
		labelvalue.foreach((Row row) -> {
			
			System.out.println(row.toString());
			double label_double = (double )row.getAs(0);
			SparseVector features =  row.getAs(1);
			DenseVector  dense_features = features.toDense();
			
			System.out.println(dense_features.toString());
			
		});

		RDD<Row> rdd_label_value = labelvalue.rdd();
		JavaRDD<Row> javardd_label_value = rdd_label_value.toJavaRDD(); 

		JavaRDD<LabeledPoint> inputData = javardd_label_value.map(
			new Function<Row, LabeledPoint> () {

				@Override
				public LabeledPoint call(Row r1) throws Exception {
					double label_double = (double )r1.getAs(0);
					SparseVector features =  r1.getAs(1);
					DenseVector  ml_dense_features = features.toDense();

					Vector mllib_vec = Vectors.dense(ml_dense_features.toArray());
       
					
					return new LabeledPoint(label_double, mllib_vec);
				}
			}
		);
			
		JavaRDD<LabeledPoint>[] tmp = inputData.randomSplit(new double[]{0.7, 0.3});
		JavaRDD<LabeledPoint> training = tmp[0]; 	// training set
		JavaRDD<LabeledPoint> test = tmp[1];     	// test set
		
		
		final NaiveBayesModel model = NaiveBayes.train(training.rdd(), 1.0);

		JavaPairRDD<Double, Double> predictionAndLabel =
  			test.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
    				@Override
    				public Tuple2<Double, Double> call(LabeledPoint p) {
      				return new Tuple2<Double, Double>(model.predict(p.features()), p.label());
    			}
  		});

		double accuracy = predictionAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
  					@Override
  					public Boolean call(Tuple2<Double, Double> pl) {
    						return pl._1().equals(pl._2());
  					}
		}).count() / (double) test.count();

		System.out.println("");
		System.out.println("... accuracy = " + String.valueOf(accuracy) );
		System.out.println("");

		model.save(sc.sc(), "target/tmp/myNaiveBayesModel");
		NaiveBayesModel sameModel = NaiveBayesModel.load(sc.sc(), "target/tmp/myNaiveBayesModel");
	
		System.out.println("naive bayes: loaded ...");
		sc.stop();
	}
	
}

