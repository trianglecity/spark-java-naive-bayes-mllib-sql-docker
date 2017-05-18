##
## Spark Java Naive Bayes for text classification
##

NOTICE 1: the sms dataset (tab separated values file) is from http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/

NOTICE 2: accuracy = 0.9148681055155875 (91%)

##

[1] download (git clone) this source coded folder.

[2] cd downloaded-source-code-folder

[3] sudo make BIND_DIR=. shell

	wait ... wait ... wait ... then a bash shell will be ready (root@9a5831cedb1f)


[4] root@9a5831cedb1f:/# cd /home/spark_MLlib

[5] root@9a5831cedb1f:/home/spark_MLlib# cd /examples/naivebayes/classification

[6] root@9a5831cedb1f:/home/spark_MLlib/examples/naivebayes/classification# sbt clean compile

[7] root@9a5831cedb1f:/home/spark_MLlib/examples/naivebayes/classification# sbt clean package


[8] root@9a5831cedb1f:/home/spark_MLlib/examples/naivebayes/classification# /spark/bin/spark-submit  --class "NBayes" ./target/scala-2.11/naivebayes_2.11-1.0.jar  ./SMSSpamCollection.txt

	
	The output looks something like this

	+-----+--------------------+--------------------+--------------------+
	|label|            sentence|               words|       filteredwords|
	+-----+--------------------+--------------------+--------------------+
	|  1.0|Go until jurong p...|[go, until, juron...|[go, jurong, poin...|
	|  1.0|Ok lar... Joking ...|[ok, lar..., joki...|[ok, lar..., joki...|
	|  0.0|Free entry in 2 a...|[free, entry, in,...|[free, entry, 2, ...|
	|  1.0|U dun say so earl...|[u, dun, say, so,...|[u, dun, say, ear...|
	|  1.0|Nah I don't think...|[nah, i, don't, t...|[nah, don't, thin...|
	|  0.0|FreeMsg Hey there...|[freemsg, hey, th...|[freemsg, hey, da...|
	|  1.0|Even my brother i...|[even, my, brothe...|[even, brother, l...|
	|  1.0|As per your reque...|[as, per, your, r...|[per, request, 'm...|
	|  0.0|WINNER!! As a val...|[winner!!, as, a,...|[winner!!, valued...|
	|  0.0|Had your mobile 1...|[had, your, mobil...|[mobile, 11, mont...|
	|  1.0|I'm gonna be home...|[i'm, gonna, be, ...|[i'm, gonna, home...|
	|  0.0|SIX chances to wi...|[six, chances, to...|[six, chances, wi...|
	|  0.0|URGENT! You have ...|[urgent!, you, ha...|[urgent!, 1, week...|
	|  1.0|I've been searchi...|[i've, been, sear...|[i've, searching,...|
	|  1.0|I HAVE A DATE ON ...|[i, have, a, date...|[date, sunday, wi...|
	|  0.0|XXXMobileMovieClu...|[xxxmobilemoviecl...|[xxxmobilemoviecl...|
	|  1.0|Oh k...i'm watchi...|[oh, k...i'm, wat...|[oh, k...i'm, wat...|
	|  1.0|Eh u remember how...|[eh, u, remember,...|[eh, u, remember,...|
	|  1.0|Fine if that?s th...|[fine, if, that?s...|[fine, that?s, wa...|
	|  0.0|England v Macedon...|[england, v, mace...|[england, v, mace...|
	+-----+--------------------+--------------------+--------------------+
	only showing top 20 rows
	
	
	+-----+--------------------+--------------------+--------------------+--------------------+
	|label|            sentence|               words|       filteredwords|         rawFeatures|
	+-----+--------------------+--------------------+--------------------+--------------------+
	|  1.0|Go until jurong p...|[go, until, juron...|[go, jurong, poin...|(100,[7,47,50,55,...|
	|  1.0|Ok lar... Joking ...|[ok, lar..., joki...|[ok, lar..., joki...|(100,[16,20,26,76...|
	|  0.0|Free entry in 2 a...|[free, entry, in,...|[free, entry, 2, ...|(100,[8,9,16,28,3...|
	|  1.0|U dun say so earl...|[u, dun, say, so,...|[u, dun, say, ear...|(100,[3,21,22,26,...|
	|  1.0|Nah I don't think...|[nah, i, don't, t...|[nah, don't, thin...|(100,[26,37,40,54...|
	|  0.0|FreeMsg Hey there...|[freemsg, hey, th...|[freemsg, hey, da...|(100,[12,14,18,19...|
	|  1.0|Even my brother i...|[even, my, brothe...|[even, brother, l...|(100,[6,23,30,47,...|
	|  1.0|As per your reque...|[as, per, your, r...|[per, request, 'm...|(100,[13,16,22,26...|
	|  0.0|WINNER!! As a val...|[winner!!, as, a,...|[winner!!, valued...|(100,[13,20,21,25...|
	|  0.0|Had your mobile 1...|[had, your, mobil...|[mobile, 11, mont...|(100,[6,10,11,24,...|
	|  1.0|I'm gonna be home...|[i'm, gonna, be, ...|[i'm, gonna, home...|(100,[6,10,12,20,...|
	|  0.0|SIX chances to wi...|[six, chances, to...|[six, chances, wi...|(100,[14,15,16,17...|
	|  0.0|URGENT! You have ...|[urgent!, you, ha...|[urgent!, 1, week...|(100,[6,9,22,34,4...|
	|  1.0|I've been searchi...|[i've, been, sear...|[i've, searching,...|(100,[10,13,16,17...|
	|  1.0|I HAVE A DATE ON ...|[i, have, a, date...|[date, sunday, wi...|(100,[12,44,61],[...|
	|  0.0|XXXMobileMovieClu...|[xxxmobilemoviecl...|[xxxmobilemoviecl...|(100,[19,23,41,43...|
	|  1.0|Oh k...i'm watchi...|[oh, k...i'm, wat...|[oh, k...i'm, wat...|(100,[26,29,42,75...|
	|  1.0|Eh u remember how...|[eh, u, remember,...|[eh, u, remember,...|(100,[25,26,27,36...|
	|  1.0|Fine if that?s th...|[fine, if, that?s...|[fine, that?s, wa...|(100,[26,30,31,49...|
	|  0.0|England v Macedon...|[england, v, mace...|[england, v, mace...|(100,[4,9,10,13,1...|
	+-----+--------------------+--------------------+--------------------+--------------------+
	only showing top 20 rows
	
	
	+-----+--------------------+
	|label|         rawFeatures|
	+-----+--------------------+
	|  1.0|(100,[7,47,50,55,...|
	|  1.0|(100,[16,20,26,76...|
	|  0.0|(100,[8,9,16,28,3...|
	|  1.0|(100,[3,21,22,26,...|
	|  1.0|(100,[26,37,40,54...|
	|  0.0|(100,[12,14,18,19...|
	|  1.0|(100,[6,23,30,47,...|
	|  1.0|(100,[13,16,22,26...|
	|  0.0|(100,[13,20,21,25...|
	|  0.0|(100,[6,10,11,24,...|
	|  1.0|(100,[6,10,12,20,...|
	|  0.0|(100,[14,15,16,17...|
	|  0.0|(100,[6,9,22,34,4...|
	|  1.0|(100,[10,13,16,17...|
	|  1.0|(100,[12,44,61],[...|
	|  0.0|(100,[19,23,41,43...|
	|  1.0|(100,[26,29,42,75...|
	|  1.0|(100,[25,26,27,36...|
	|  1.0|(100,[26,30,31,49...|
	|  0.0|(100,[4,9,10,13,1...|
	+-----+--------------------+
	only showing top 20 rows
	
	... accuracy = 0.9148681055155875
	

[9] root@9a5831cedb1f:/home/spark_MLlib/examples/naivebayes/classification/target/tmp# rm -rf ./myNaiveBayesModel/	

##

[10] the source code looks like this

	
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

