# Reproducibility and Testing

The simple test script **test-project.sh** runs Python script **project** twice and uses `diff` to compare the 
intermediate results saved in files like **03a_data_x_train.json**. For this to have any chance to work, all 
non-deterministc algorithms must be disabled. Currently this is done by this code snippet:

```python
if config["alpha_vantage"]["mode"] == "test":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
```

The background of this is described chapter [Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html#reproducibility) and [torch.use_deterministic_algorithms](https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms) in the PyTorch documentation.

It took me quite a while to fully understand the consequences of the first sentence in 
chapter [Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html#reproducibility)

> Completely reproducible results are not guaranteed across PyTorch releases, individual commits, 
> or different platforms. Furthermore, results may not be reproducible between CPU and GPU 
> executions, even when using identical seeds.

**This means, when running my little test script on another machine the tests will most probably fail.**

For more insight please refer to the discussion in [Clearly different results on different machines](https://discuss.pytorch.org/t/clearly-different-results-on-different-machines/81768).

On [stackoverflow](https://stackoverflow.com/) I found a question about [PyTorch VERY different results on different machines using docker and CPU](https://stackoverflow.com/questions/64240440/pytorch-very-different-results-on-different-machines-using-docker-and-cpu). As apossible solution to the problem at hand is this code snippet:

```python
    np.random.seed(42)
    torch.manual_seed(42)
    os.environ["PYTHONHASHSEED"] = "42"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
```

So, I changed the first code snippet above to

```python
if config["alpha_vantage"]["mode"] == "test":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    os.environ["PYTHONHASHSEED"] = "42"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
```

and re-ran the test script. As expected, it failed because all "05a...." and above files differ from the sanctioned 
outputs. To test the stackoverflow solution, I updated the sanctioned outputs with the files generated by the test script 
using the new version of the code (project.py at commit 101e44a14c56c9cca2a6882d2f575dd4f9326cf1).

**However, I'm quite sure that we need a different approach to testing ...**

