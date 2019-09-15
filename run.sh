docker run -v ${PWD}/result:/usr/local/app/result fraud:1.0.0 python src/main.py train.csv test.csv "result/submission.csv" > result/logs.txt
