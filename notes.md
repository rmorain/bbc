# Need to allow for the base models tokenizer
    - To do this, you should probably convert prefix to a string before evaluating it.

# Pandas Dataframe
- Store prefix, prompt, continuation
- Log as table

# Validation
- Run validation on a trained model
- Log stats

# Log entropy loss

# Royal Energy
- This doesn't look like its working.
- Total loss is low with some spikes from the policy loss.
- Is it because I am not using seed 0?
- Should I try to recreate results exactly? 
- Maybe it just needs more time.

# Evaluate
- What should this look like?
- Run test on a list of datasets
- Produce a table
- Is this experiment specific?
    - Maybe
    - Are there common elements?
        - Test dataset
        - Table
        - Metrics
            - Accuracy
            - Perplexity
            - Reward
            - Diversity
- For sentiment, I change the target. 
    - Is this a new dataset?
        - Yes
        - Kinda
        - In order to be general, you can just treat it like a new dataset, rather
            than a change in target.
        - This is kind of good because then I can isolate the testing better. 

# Train and evaluate
- Can I train using `train.py` and then evaluate using `evaluate.py`?
- Should I use the same `accelerator` object?
    - Pass in `ppo_trainer`
- Should I want to train and evaluate in the same job?

# Config via terminal flags

# Run sentiment control experiment fully
- Test evaluation with multiple GPUs
    - Should work?

# Add confidence
- Need to add confidence to the metrics
- How to measure this?

```python
perplexities = test_df.loc[(test_df["target"] == 1)]["perplexity"]

std = perplexities.std()

p = perplexities.sum() / len(perplexities)
z = 1.960  # 95% confidence interval

margin_of_error = z * (std / math.sqrt(len(perplexities)))
print(margin_of_error)
```

Given a list of values, 
    1. compute the standard deviation of these values
    2. Margin of error = z * std / sqrt(n)
        - z is the confidence interval
            - The mean is a certain value with CI probability:w
        - Bigger the margin of error, the wider the range that the mean could be
        - CI
            - How often the interval would contain the true population parameter if the
                experiment were conducted several times

# Add a wandb description
- Describe what the run is for. What makes it special?
- Done

# Identify differences
- Unbalanced dataset
- compute reward
    - Only use continuation, not prompt
- Scores/whitening/normalization
- Num epochs
    - Train longer?
- GPT-2 large
    - Base model
- Increase training context window
    - Compute average context window length
        Mean prompt length:  9.0476
        STD of prompt length:  3.86177863684598
        Range 5-13 tokens

# Decrease memory utilization by 50%
- Only using 20%

# Why is perplexity 0?

# Simplify evaluate config

# Check out eval logs
Look at /home/rmorain2/bbc/local_logs/tsqkp0ir/positive_prompts_neg_log_tsqkp0ir.csv
    - Is this correct?
    - How to fix?
    - The dataset is not being split across the processes.
        - Add dataloader to accelerator?

# Processed dataset
- Labels are all wrong?
    - All negative
    - Why?

# Training/Eval time
- Log the time to train/eval

# Fix test table 
- Compute diversity on a per prompt level (Every 25)

# Maximize evaluation batch size
- Need to speed this up

# Tuning learning rate
- 1e-4 too big
- Try 1e-5

# Llama 3.1 8B
- Debug this
- Timeout error
- May not work in parallel for some reason
- Might be taking a long time because of the batch size

# Infinite training
- Make job preemptible or restartable
- Decouple train and evaluate
- Thoughts and questions
    - Apparently, you cannot resume an offline run with wandb. Therefore, when a job is 
        restarted it will have to be a new wandb run because running an online run is 
        impossible. 
    - What does it look like to restart a run?
        - When first starting, everything will look the same. Load the specified model,
            and train normaly.
        - When a signal interrupts training, save the model.
        - When the job is restarted, load the saved model and continue training where
            you left off. 
    - Questions
        - How do you handle a signal that interrupts the job?
        - When a job is restarted, how do you know to load a saved model? How do you 
            know the job has been restarted? 
            - I need to pass the path to the saved model in the new job script. 
                - No I don't. I can check if a saved model already exists and use the 
                    latest one. I might have to use the slurm job id in the saved models
                    directory.
            - How do I run this restart script?
                - `scontrol requeue <jobid>`
            - Will it have a different job id?
                - No
            - I need access to the old script and add the new model path
                - The same parameters as before will be run, but if you changed the 
                    training script then the new training script will be run. You need
                    to create a new training script for each experiment now.
        - When a job gets interrupted, do I need to pick up where I left off? How do I
            do this?
            - I need to look at the `accelerate` docs for this I think.