require('init');

tester = torch.Tester()

tester:add(require('test.Linear'))

tester:add(require('test.TransferFunctionALU'))
tester:add(require('test.TransferFunctionReCU'))
tester:add(require('test.TransferFunctionFactory'))

tester:add(require('test.NetworkSequential'))
tester:add(require('test.NetworkScalarFunction'))

tester:add(require('test.ErrorCriterionSquare'))
tester:add(require('test.ErrorCriterionSquareRelative'))
tester:add(require('test.ErrorCriterionRelative'))

tester:add(require('test.ApproximationNormalizatorFactory'))

tester:run()