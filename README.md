# Neural Network Sequence Trainer

Neural network trainer for sequences build on top of [torch/nn](https://github.com/torch/nn).

## Dependencies

Assuming that you have [torch](http://torch.ch/) already installed, you will also need:

* [inspect](https://github.com/kikito/inspect.lua)
* [dkjson](http://dkolf.de/src/dkjson-lua.fsl/home)
* [moses](https://github.com/Yonaba/Moses)

## Tests

Unit tests

```bash
th test.lua
```

## Developer's documentation

Some useful documentation:

### Neural Networks

* Vanishing gradients issue, see <https://cs224d.stanford.edu/notebooks/vanishing_grad_example.html>.

### Torch and runtime environment

* Torch manual, see <https://github.com/torch/torch7/blob/master/doc/tensor.md>.
* How to develop new module, see <http://torch.ch/docs/developer-docs.html>.
* Unit tests, see <https://github.com/torch/torch7/blob/master/doc/tester.md>
* THNN api, see <https://github.com/torch/nn/blob/master/lib/THNN/doc/api_reference.md>.
* Introduction to 'nn' package, see <https://github.com/torch/nn/blob/master/doc/index.md>.
* Gnuplot, see <https://github.com/torch/gnuplot/blob/master/doc/plotline.md#gnuplot.line.dok>.