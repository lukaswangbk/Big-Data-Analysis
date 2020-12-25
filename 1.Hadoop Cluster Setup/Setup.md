# Hadoop Cluster Setup
Hadoop is an open-source software framework used for distributed storage and processing. We are going to setup a Hadoop cluster using Google
Compute Engine. The official tutorial for Amazon EC2 can be found [here](https://aws.amazon.com/getting-started).

## Task 1: Single-node Hadoop Setup
**Step 1:** Linux VM setup
          
    Follow the official instruction and open the console.
[official Instruction](https://cloud.google.com/compute/docs/quickstart-linux) 

**Step 2:** Update the OS system
```Bash
sudo apt-get update
```
**Step 3:** Install Java 8
```Bash
sudo apt-get update
sudo apt-get install default-jre
sudo apt-get install default-jdk
```
**Step 4:** Download hadoop 2.9.2
```Bash
wget http://apache.cs.utah.edu/hadoop/common/hadoop-2.9.2/hadoop-2.9.2.tar.gz
```
**Step 5:** Download and untar hadoop 2.9.2
```Bash
wget http://apache.cs.utah.edu/hadoop/common/hadoop-2.9.2/hadoop-2.9.2.tar.gz
ls -ltr|grep -i hadoop
tar -xvzf hadoop-2.9.2.tar.gz
mv hadoop-2.9.2 hadoop2 
```
**Step 6:** Modify `.bashrc`
```Bash
vi .bashrc
```
```Bash
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
export HADOOP_INSTALL=/home/g295334279/hadoop2
export PATH=$PATH:$HADOOP_INSTALL/bin
export PATH=$PATH:$HADOOP_INSTALL/sbin
export HADOOP_MAPRED_HOME=$HADOOP_INSTALL
export HADOOP_COMMON_HOME=$HADOOP_INSTALL
export HADOOP_HDFS_HOME=$HADOOP_INSTALL
export YARN_HOME=$HADOOP_INSTALL
export HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_INSTALL/lib/native
export HADOOP_OPTS="-Djava.library.path=$HADOOP_INSTALL/lib"
export HADOOP_CONF_DIR=$HADOOP_INSTALL/etc/hadoop
export HADOOP_CLASSPATH=${JAVA_HOME}/lib/tools.jar
export YARN_CONF_DIR=$HADOOP_INSTALL/etc/hadoop
export PATH=$PATH:$HADOOP_CONF_DIR/bin
export PATH=$PATH:$YARN_CONF_DIR/sbin
```
**Step 7:** Update `.bashrc`
```Bash
. .bashrc
```
**Step 8:** Configure `hadoop-config.sh`
```Bash
vim /home/g295334279/hadoop2/libexec/hadoop-config.sh
```
```Bash
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
```
**Step 9:** Configure `yarn-env.sh`
```Bash
vim /home/g295334279/hadoop2/etc/hadoop/yarn-env.sh
```
```Bash
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
export HADOOP_HOME=/home/g295334279/hadoop2
export HADOOP_MAPRED_HOME=$HADOOP_HOME
export HADOOP_COMMON_HOME=$HADOOP_HOME
export HADOOP_HDFS_HOME=$HADOOP_HOME
export YARN_HOME=$HADOOP_HOME
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
export YARN_CONF_DIR=$HADOOP_HOME/etc/hadoop
export PATH=$PATH:$HADOOP_HOME/bin
```
**Step 10:** Configure `hadoop-env.sh`
```Bash
vim /home/g295334279/hadoop2/etc/hadoop/hadoop-env.sh
```
```Bash
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
export HADOOP_INSTALL=/home/g295334279/hadoop2
export PATH=$PATH:$HADOOP_INSTALL/bin
export PATH=$PATH:$HADOOP_INSTALL/sbin
export HADOOP_MAPRED_HOME=$HADOOP_INSTALL
export HADOOP_COMMON_HOME=$HADOOP_INSTALL
export HADOOP_HDFS_HOME=$HADOOP_INSTALL
export HADOOP_CLASSPATH=${JAVA_HOME}/lib/tools.jar
export YARN_HOME=$HADOOP_INSTALL
export HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_INSTALL/lib/native
export HADOOP_OPTS="-Djava.library.path=$HADOOP_INSTALL/lib"
```
**Step 11:** Configure `core-site.xml`
```Bash
vim /home/spark/hadoop2/etc/hadoop/core-site.xml
```
```xml
<property>
<name>fs.default.name</name>
<value>hdfs://instance-1:9000</value> 
<final>true</final>
</property>
 <property>
<name>hadoop.proxyuser.hadoop.hosts</name>
<value>localhost</value>
</property>
<property>
<name>hadoop.proxyuser.hadoop.groups</name>
<value>users</value>
</property>
<property>
<name>hadoop.tmp.dir</name>
<value>/home/g295334279/hdfsdrive</value>
</property>
```
**Step 12:** Configure `mapred-site.xml`
```Bash
cp /home/g295334279/hadoop/etc/hadoop/mapred-site.xml.template /home/g295334279/hadoop/etc/hadoop/mapred-site.xml
vi /home/g295334279/hadoop/etc/hadoop/mapred-site.xml
```
```xml
<property>
<name>mapreduce.framework.name</name>
<value>yarn</value>
</property> 

<property>
<name>mapred.child.java.opts</name>
<value>-Xmx1200m</value>
</property>
```
**Step 13:** Create directories for output (Optional)
```Bash
mkdir /home/g295334279/hdfsdrive
```
hdfs-site.xml
**Step 14:** Configure `hdfs-site.xml`
```Bash
vi /home/g295334279/hadoop2/etc/hadoop/hdfs-site.xml
```
```xml
<property>
<name>dfs.replication</name>
<value>1</value>
</property>

<property>
<name>dfs.permissions</name>
<value>false</value>
</property>
```
**Step 15:** Configure `yarn-site.xml`
```Bash
vi $HADOOP_CONF_DIR/yarn-site.xml
```
```xml
<property>
<name>yarn.nodemanager.aux-services</name>
<value>mapreduce_shuffle</value>
</property>
<property>
<name>yarn.nodemanager.aux-services.mapreduce.shuffle.class</name>
<value>org.apache.hadoop.mapred.ShuffleHandler</value>
</property>
```
**Step 16:** Format namenode
```Bash
hadoop namenode -format
```
**Step 17:** Start hadoop
```Bash
hadoop-daemon.sh start namenode
hadoop-daemons.sh start datanode
yarn-daemon.sh start resourcemanager
yarn-daemons.sh start nodemanager
```
**Step 18:** Stop hadoop
```Bash
hadoop-daemon.sh start namenode
hadoop-daemons.sh start datanode
yarn-daemon.sh start resourcemanager
yarn-daemons.sh start nodemanager
```
**Step 19:** Standalone operation 
```Bash
bin/hdfs dfs -mkdir -p input
bin/hdfs dfs -ls input
bin/hadoop jar share/hadoop/mapreduce/hadoop-mapreduce-examples-2.9.2.jar grep input output 'dfs[a-z.]+'
```
**Step 20:** Setup passphraseless ssh
```Bash
cd ~/.ssh
ssh-keygen
id_rsa
cp id_rsa.pub authorized_keys
authorized_keys  id_rsa  id_rsa.pub  known_hosts
ssh localhost
```
**Step 21:** Execution
```Bash
bin/hdfs namenode -format
sbin/start-dfs.sh
```
**Step 22:** Tesing the service
```Bash
jps
```

    Check on port 50070 with external IP address
**Step 23:** Check with Terasort
```Bash
./bin/hadoop jar ./share/hadoop/mapreduce/hadoop-mapreduce-examples-2.9.2.jar teragen 100000 terasort/input
./bin/hadoop jar ./share/hadoop/mapreduce/hadoop-mapreduce-examples-2.9.2.jar terasort terasort/input terasort/output
./bin/hadoop jar ./share/hadoop/mapreduce/hadoop-mapreduce-examples-2.9.2.jar teravalidate terasort/output terasort/check
```
**Step 24:** Congrats and play around with it

## Task 2: Multiple-node Hadoop Setup 
### 4VM and 3 Slaves
:heavy_exclamation_mark: **Following steps should be implemented on all slaves**

**Step 1:** Linux VMs setup

    Follow the official instruction and open the console.
[official Instruction](https://cloud.google.com/compute/docs/quickstart-linux)

**Step 2:** Add hduser
```Bash
sudo -i
adduser hduser
passwd hduser
```
**Step 3:** Configure `sshd_config`
```Bash
vim /etc/ssh/sshd_config
```
```Bash
PasswordAuthentication yes
ChallengeResponseAuthentication yes
```
**Step 4:** Restart sshd
```Bash
service sshd restart
```
**Step 5:** Get IP address and fully qualified domain name for all VMs
```Bash
hostname -i
hostname -f
```
**Step 6:** Configure `hosts` to connect all instances
```Bash
vim /etc/hosts
```

    Add IP address and fully qualified domain name for all other VMs get from Step 5

**Step 7:** Generate public private key using rsa
```Bash
su hduser
cd ~
ssh-keygen -t rsa -P ""
```
**Step 8:** Copy public private key to other instances
```Bash
ssh-copy-id -i /home/hduser/.ssh/id_rsa.pub hduser@instance-2
ssh-copy-id -i /home/hduser/.ssh/id_rsa.pub hduser@instance-3
ssh-copy-id -i /home/hduser/.ssh/id_rsa.pub hduser@instance-4
```
**Step 9:** Authorized keys
```Bash
chmod 0600 ~/.ssh/authorized_keys
```
**Step 10:** Connection testing
```Bash
ssh instance-2
exit
```
**Step 11:** Configure `hdfs-site.xml`
```xml
<configuration>
<property>
<name>dfs.replication</name>
<value>3</value>
<description>Default block replication.
The actual number of replications can be specified when the file is created.
The default is used if replication is not specified in create time.
</description>
</property>
<property>
<name>dfs.namenode.name.dir</name>
<value>file:/home/g295334279/hadoop_store/hdfs/namenode</value>
</property>
<property>
<name>dfs.datanode.data.dir</name>
<value>file:/home/g295334279/hadoop_store/hdfs/datanode</value>
</property>
<property>
<name>dfs.namenode.checkpoint.dir</name>
<value>file:/home/g295334279/hadoop_store/hdfs/secondarynamenode</value>
</property>
<property>
<name>dfs.namenode.checkpoint.period</name>
<value>3600</value>
</property>
</configuration>
```
**Step 12:** Configure `core-site.xml`
```xml
<configuration>

<property>
<name>fs.default.name</name>
<value>hdfs://instance-1:9000</value>  <!--replace hadoop1 to your hostname-->
<final>true</final>
</property>

<property>
<name>hadoop.proxyuser.hadoop.hosts</name>
<value>localhost</value>
</property>

<property>
<name>hadoop.proxyuser.hadoop.groups</name>
<value>users</value>
</property>

<property>
<name>hadoop.tmp.dir</name>
<value>/home/g295334279/hadoop_store/tmp</value>
<description>A base for other temporary directories.</description>
</property>

</configuration>
```
**Step 12:** Configure `mapred-site.xml`
```xml
<configuration>

<property>
<name>mapreduce.framework.name</name>
<value>yarn</value>
</property> 

<property>
<name>mapred.child.java.opts</name>
<value>-Xmx1200m</value>
</property>

<property>
<name>mapreduce.job.reduces</name>
<value>30</value>
</property>

<property>
<name>mapreduce.reduce.memory.mb</name>
<value>4096</value>
</property>

<property>
<name>mapred.job.tracker</name>
<value>instance-1:9001</value>
<description>The host and port that the MapReduce job tracker runs
at. If "local", then jobs are run in-process as a single map
and reduce task.
</description>
</property>

</configuration>
```
**Step 12:** Configure `yarn-site.xml`
```xml
<configuration>

<!-- Site specific YARN configuration properties -->
<property>
    <name>yarn.resourcemanager.hostname</name>
    <value>instance-1</value>
</property>
<property>
    <name>yarn.nodemanager.aux-services</name>
    <value>mapreduce_shuffle</value>
</property>
<property>
    <name>yarn.nodemanager.aux-services.mapreduce.shuffle.class</name>
    <value>org.apache.hadoop.mapred.ShuffleHandler</value>
</property>

</configuration>
```
**Step 13:** Open `.bashrc` and setup the dependency and environment variables
```Bash
su hduser
vi /home/hduser/.bashrc
```
```Bash
export HADOOP_PREFIX=/home/g295334279/hadoop2
export HADOOP_HOME=/home/g295334279/hadoop2
export HADOOP_MAPRED_HOME=${HADOOP_HOME}
export HADOOP_COMMON_HOME=${HADOOP_HOME}
export HADOOP_HDFS_HOME=${HADOOP_HOME}
export YARN_HOME=${HADOOP_HOME}
export HADOOP_CONF_DIR=${HADOOP_HOME}/etc/hadoop
# Native Path
export HADOOP_COMMON_LIB_NATIVE_DIR=${HADOOP_PREFIX}/lib/native
export HADOOP_OPTS="-Djava.library.path=$HADOOP_PREFIX/lib"
export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
export JRE_HOME=/usr/lib/jvm/java-8-openjdk-amd64/jre
export PATH=$PATH:/usr/lib/jvm/java-8-openjdk-amd64/bin:/usr/lib/jvm/java-8-openjdk-amd64/jre/bin
```
**Step 14:** Update the bash
```Bash
source ~/.bashrc
```
**Step 15:** Set environment for Slaves

    Install Java 8 as Step 2 in Single-node Cluster Setup

```Bash
cd /home/g295334279

scp -r hadoop2 instance-2:/home/hduser/
scp -r hadoop_store instance-2:/home/hduser/
scp -r /home/hduser/.bashrc instance-2:/home/hduser/
scp -r hadoop2 instance-3:/home/hduser/
scp -r hadoop_store instance-3:/home/hduser/
scp -r /home/hduser/.bashrc instance-3:/home/hduser/
scp -r hadoop2 instance-4:/home/hduser/
scp -r hadoop_store instance-4:/home/hduser/
scp -r /home/hduser/.bashrc instance-4:/home/hduser/

sudo -i
mv /home/hduser/hadoop_store /home/g295334279
mv /home/hduser/hadoop2 /home/g295334279
chown -R hduser:hduser /home/g295334279/hadoop2
chown -R hduser:hduser /home/g295334279/hadoop_store
```
**Step 16:** Change slaves
```Bash
cd $HADOOP_CONF_DIR
vi slaves
```
```Bash
#localhost
instance-1
instance-2
instance-3
Instance-4
```
**Step 17:** Format and start the service
```Bash
hadoop namenode -format
start-dfs.sh
start-yarn.sh
```
**Step 18:** Testing the service
```Bash
jps
ssh instance-2
jps
ssh instance-3
jps
ssh instance-4
jps
```

    Check on port 50070/ 8088 (if available) with external IP address
**Step 19:** Check with 2GB Terasort
```Bash
hadoop jar ./share/hadoop/mapreduce/hadoop-mapreduce-examples-2.9.2.jar teragen 21474837 terasort/2G-input
hadoop jar ./share/hadoop/mapreduce/hadoop-mapreduce-examples-2.9.2.jar terasort terasort/2G-input terasort/2G-output
```
**Step 20:** Check with 2GB Terasort
```Bash
hadoop jar ./share/hadoop/mapreduce/hadoop-mapreduce-examples-2.9.2.jar teragen 214748365 terasort/20G-input
hadoop jar ./share/hadoop/mapreduce/hadoop-mapreduce-examples-2.9.2.jar terasort terasort/2G-input terasort/20G-output
```
## Task 3: Python 2 and Java Job Comparasion
### Dataset Preparation
```Bash
sudo -i
wget https://www.dropbox.com/s/8nlgr2ilum9elb7/shakespeare.zip?dl=0
unzip MapReduce_WordCount.zip?dl=0
chown -R hduser:hduser /home/g295334279/MapReduce_WordCount
chmod +x /home/g295334279/MapReduce_WordCount/mapper.py
chmod +x /home/g295334279/MapReduce_WordCount/reducer.py 
chmod +x /home/g295334279/MapReduce_WordCount/WordCount.java
exit
su hduser
wget https://www.dropbox.com/s/8nlgr2ilum9elb7/shakespeare.zip?dl=0
unzip shakespeare.zip?dl=0
mkdir shakespeare_output
mkdir shakespeare_input
mv shakespeare-basket* shakespeare_input
```
### Python 2 Mapper & Reducer
```Bash
./bin/hadoop jar ./share/hadoop/tools/lib/hadoop-streaming-2.9.2.jar -file MapReduce_WordCount/mapper.py -mapper mapper.py -file MapReduce_WordCount/reducer.py -reducer reducer.py -input /user/hduser/input/* -output /user/hduser/output
```
### Java Mapper & Reducer
```Bash
bin/hadoop com.sun.tools.javac.Main MapReduce_WordCount/WordCount.java
cd MapReduce_WordCount/
jar cf wc.jar org.apache.hadoop.examples.WordCount*.class
./bin/hadoop jar MapReduce_WordCount/wc.jar  org.apache.hadoop.examples.WordCount input java_output
```
### Results
Language | Map Time | Reduce Time 
--- | --- | --- 
JAVA | 57106ms | 138405ms
Python | 114227ms | 436342ms

*Java uses less time compared with Python both in map and reduce tasks.*
