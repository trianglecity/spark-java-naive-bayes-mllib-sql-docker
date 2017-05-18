name := "naivebayes"
version := "1.0"
scalaVersion := "2.11.8"
javacOptions ++= Seq("-source", "1.8", "-target", "1.8")
libraryDependencies ++= Seq(
		"commons-io" % "commons-io" % "2.5",
		"log4j" % "log4j" % "1.2.17",
		"org.apache.spark" %% "spark-core" % "2.1.2-SNAPSHOT",
  		"org.apache.spark" %% "spark-sql" % "2.1.2-SNAPSHOT",
  		"org.apache.spark" %% "spark-mllib" % "2.1.2-SNAPSHOT" ,
  		"org.apache.spark" %% "spark-streaming" % "2.1.2-SNAPSHOT" 
		)


resolvers += "Local Maven Repository" at "file:///root/.m2/repository"

autoScalaLibrary := false
