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