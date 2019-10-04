docker run -v ${PWD}/result:/usr/local/app/result fraud:1.0.1 python src/main.py train.csv test.csv "result/cv_results.csv" "result/submission.csv" > result/logs.txt
