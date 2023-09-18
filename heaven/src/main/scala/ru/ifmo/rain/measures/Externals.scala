package ru.ifmo.rain.measures

import org.apache.spark.sql.DataFrame
import ru.ifmo.rain.measures.ExtConjugacy.comb
import ru.ifmo.rain.{CLASS_COL, ID_COL, LABEL_COL, Sparkling}


/**
 * Computes necessary values to evaluate any external measure
 */
object Externals {
  private val EQ = "="
  private val NOT_EQ = "!="

  private def pairwiseStat(df: DataFrame, classComp: String, labelComp: String): Long = df.sqlContext.sql(
    f"""SELECT count(*) FROM ext_df AS df_1 CROSS JOIN ext_df AS df_2 WHERE df_1.$ID_COL < df_2.$ID_COL
          and df_1.$CLASS_COL $classComp df_2.$CLASS_COL and df_1.$LABEL_COL $labelComp df_2.$LABEL_COL
    """.stripMargin).first().getLong(0)

  /**
   * Calculates TP, FP, FN, TN values for given dataframe with both labels and external classes
   * @param df Multimodal dataframe
   * @return External measures' values
   */
  @Sparkling
  def pairwise(df: DataFrame): ExtPairwise = {
    val n = comb(df.count())
    df.createOrReplaceTempView("ext_df")
    val tp = pairwiseStat(df, classComp = EQ, labelComp = EQ)
    val fp = pairwiseStat(df, classComp = NOT_EQ, labelComp = EQ)
    val fn = pairwiseStat(df, classComp = EQ, labelComp = NOT_EQ)
    val tn = n - (tp + fp + fn)
    new ExtPairwise(tp / n.toDouble, fp / n.toDouble, fn / n.toDouble, tn / n.toDouble)
  }

  /**
   * Calculates conjugacy matrix for given dataframe with both labels and external classes
   * @param df Multimodal dataframe
   * @return External measures' values
   */
  @Sparkling
  def conjugate(df: DataFrame): ExtConjugacy = {
    val matrix = df.stat.crosstab(CLASS_COL, LABEL_COL)
      .drop(CLASS_COL + "_" + LABEL_COL).collect()
      .map { _.toSeq.map { _.asInstanceOf[Long] }.toArray }
    new ExtConjugacy(matrix)
  }
}
