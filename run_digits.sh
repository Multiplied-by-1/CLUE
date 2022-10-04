python main.py --dataset digits --method CLUE --epochs 20 --batch_size 128 --output res/clue_digits.log
python main.py --dataset digits --method Naive --epochs 20 --batch_size 128 --output res/naive_digits.log
python main.py --dataset digits --method joint_training --epochs 20 --batch_size 128
python main.py --dataset digits --method Replay --epochs 20 --batch_size 128 --output res/replay_digits.log
