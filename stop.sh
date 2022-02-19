kill -9  `ps -ef|grep train.sh|awk '{print $2}'`
kill -9  `ps -ef|grep train.py|awk '{print $2}'`
ps -ef|grep train.py