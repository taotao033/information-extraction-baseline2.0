nohup python -u train.py --embeddings bert768 --embedding_size 768 --max_sentence_length 100 --batch_size 64 --test_batch_size 32 --num_epochs 100 > log 2>&1 & tail -f log 
