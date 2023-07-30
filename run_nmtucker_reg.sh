

for lambda in 1e-7 1e-6
do

#python nmtucker_eval.py -model=ML1 -dataset=fun -core_shape=10,10,10 -num_experiments=2 -regularization=L1 -lambda_l1=$lambda
#python nmtucker_eval.py -model=ML1 -dataset=fun -core_shape=20,20,20 -num_experiments=2 -regularization=L1 -lambda_l1=$lambda
#python nmtucker_eval.py -model=ML1 -dataset=fun -core_shape=40,40,40 -num_experiments=2 -regularization=L1 -lambda_l1=$lambda

python nmtucker_eval.py -model=ML1 -dataset=eps -core_shape=10,10,10 -num_experiments=2 -regularization=L1 -lambda_l1=$lambda
python nmtucker_eval.py -model=ML1 -dataset=eps -core_shape=20,20,20 -num_experiments=2 -regularization=L1 -lambda_l1=$lambda
python nmtucker_eval.py -model=ML1 -dataset=eps -core_shape=40,40,40 -num_experiments=2 -regularization=L1 -lambda_l1=$lambda

#python nmtucker_eval.py -model=ML1 -dataset=chars -core_shape=40,40,40 -num_experiments=2 -regularization=L1 -lambda_l1=$lambda
#python nmtucker_eval.py -model=ML1 -dataset=chars -core_shape=80,80,80 -num_experiments=2 -regularization=L1 -lambda_l1=$lambda

done