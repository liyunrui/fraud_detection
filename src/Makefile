null-importance:
		python main.py --train_file ../../dataset/train.csv --test_file ../../dataset/test.csv --result_path ../result/submission_null.csv --feature_selection True --TEST_NULL_HYPO True
train-lgb:
		python main.py --train_file ../../dataset/train.csv --test_file ../../dataset/test.csv --result_path ../result/submission_lgb.csv 
train-lgb-fs:
		python main.py --train_file ../../dataset/train.csv --test_file ../../dataset/test.csv --result_path ../result/submission_lgb.csv --feature_selection True
train-xgb:
		python main.py --train_file ../../dataset/train.csv --test_file ../../dataset/test.csv --result_path ../result/submission_xgb.csv --model xgb 
mf:
		python matrix_factorization.py --train_file ../../dataset/train.csv --test_file ../../dataset/test.csv --latent_feature_path ../features
mf-mchno:
		python matrix_factorization.py --train_file ../../dataset/train.csv --test_file ../../dataset/test.csv --latent_feature_path ../features --column_name mchno
mf-contp:
		python matrix_factorization.py --train_file ../../dataset/train.csv --test_file ../../dataset/test.csv --latent_feature_path ../features --column_name contp
mf-scity:
		python matrix_factorization.py --train_file ../../dataset/train.csv --test_file ../../dataset/test.csv --latent_feature_path ../features --column_name scity
mf-stocn:
		python matrix_factorization.py --train_file ../../dataset/train.csv --test_file ../../dataset/test.csv --latent_feature_path ../features --column_name stocn
mf-cano-mchno:
		python matrix_factorization.py --train_file ../../dataset/train.csv --test_file ../../dataset/test.csv --latent_feature_path ../features --row_name cano --column_name mchno
mf-mchno-loctm_hour_of_day:
		python matrix_factorization.py --train_file ../../dataset/train.csv --test_file ../../dataset/test.csv --latent_feature_path ../features --row_name mchno --column_name loctm_hour_of_day
mf-cano-locdt:
		python matrix_factorization.py --train_file ../../dataset/train.csv --test_file ../../dataset/test.csv --latent_feature_path ../features --row_name cano --column_name locdt
