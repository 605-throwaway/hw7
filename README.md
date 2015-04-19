# hw7
This can be run on AWS EMR using the command:
python2.7 eval2.pyc eval_acc.log spark/bin/spark-submit dsgd_mf.py 20 3 3 0.9 0.1 s3://path_to_input.csv w.csv h.csv

If the number of iterations is to high (above 4 or 5) it will take a very long time to run.  
With 3 iterations it ran in under a minute.
