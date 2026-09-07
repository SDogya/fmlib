import os
import sys
from pyspark.sql import SparkSession
from pyspark import SparkConf


def get_conf(is_datalab_pro):
    if is_datalab_pro:
        _CONF = SparkConf().setAll([
            ("spark.ui.enabled", "true"),
            ("spark.executor.cores", "10"),
            ("spark.kubernetes.executor.limit.cores", "10"),
            ("spark.kubernetes.executor.request.cores", "10"),
            ("spark.executor.memory", "20g"),
            ("spark.executor.memoryOverhead", "20g"),
            ("spark.sql.parquet.compression.codec", "snappy"),
            ("spark.dynamicAllocation.enabled", "true"),
            ("spark.dynamicAllocation.minExecutors", "0"),
            ("spark.dynamicAllocation.maxExecutors", "8"),
            ("spark.dynamicAllocation.executorIdleTimeout", "300s"),
            ("spark.dynamicAllocation.schedulerBacklogTimeout", "5s"),
            ("spark.driver.memory", "20g"),
            ("spark.dynamicAllocation.cachedExecutorIdleTimeout", "1800s"),
            ('spark.sql.execution.arrow.pyspark.enabled', 'true'),
            ("spark.eventLog.enabled", "false"),
            ("spark.driver.maxResultSize", "10G")
        ])
    else:
        _CONF = SparkConf().setAll([
            ("spark.local.dir", "sparktmp"),
            ("spark.port.maxRetries", "64"),
            ("spark.serializer", "org.apache.spark.serializer.KryoSerializer"),
            ("spark.kryoserializer.buffer.max", "1g"),
            ("spark.executor.cores", "4"),
            ("spark.executor.memory", "17G"),
            ("spark.dynamicAllocation.enabled", "true"),
            ("spark.dynamicAllocation.minExecutors", "0"),
            ("spark.dynamicAllocation.maxExecutors", "12"),
            ("spark.dynamicAllocation.initialExecutors", "4"),
            ("spark.dynamicAllocation.executorIdleTimeout", "120s"),
            ("spark.dynamicAllocation.schedulerBacklogTimeout", "10"),
            ("spark.driver.memory", "15G"),
            ("spark.memory.fraction", "0.8"),
            ("spark.sql.hive.caseSensitiveInferenceMode", "NEVER_INFER"),
            ("spark.sql.hive.metastorePartitionPruning", "true"),
            ("spark.network.timeout", "60"),
            ("spark.driver.maxResultSize", "15G"),
            ('spark.sql.shuffle.partitions', '1000'),
            ("hive.exec.dynamic.partition", "true"),
            ("hive.exec.dynamic.partition.mode", "nonstrict"),
            ("spark.sql.sources.partitionOverwriteMode", "dynamic"),
            ("spark.dynamicAllocation.cachedExecutorIdleTimeout", "60s"),
            ("spark.sql.execution.arrow.pyspark.enabled", "false"),
            ("spark.sql.hive.convertMetastoreParquet", "false"),
            ("mapred.input.dir.recursive", "true"),
            ("hive.mapred.supports.subdirectories", "true"),
            ("spark.sql.autoBroadcastJoinThreshold", "-1"),
            ("spark.sql.hive.manageFilesourcePartitions", "false"),
            ("spark.shuffle.service.enabled", "true"),
            ("spark.sql.catalogImplementation", "hive"),
            ('spark.sql.execution.arrow.pyspark.enabled', 'true')
        ])
    return _CONF

_CONF = get_conf(is_datalab_pro=False)

def get_spark(conf=_CONF, app_name='srs-spark'):
    spark = SparkSession \
        .builder \
        .appName(app_name) \
        .config(conf=conf) \
        .getOrCreate()
    return spark