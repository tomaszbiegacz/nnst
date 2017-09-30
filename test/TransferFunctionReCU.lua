local tests = torch.TestSuite()

local precision = 1e-3

function tests.TransferFunctionReCU_testOutput()
    local tr = nnst.TransferFunctionReCU()

    local xs = torch.Tensor{-5, -1, -0.1, 0, 1, 5}
    local result = tr:forward(xs)
    tester:eq(torch.Tensor{0, 0, 0, 0, 1, 125}, result, precision)    
end

function tests.TransferFunctionReCU_testGradient()
    local tr = nnst.TransferFunctionReCU()

    local xs = torch.Tensor{-5, -1, -0.1, 0, 1, 5}
    tr:forward(xs)

    local go = torch.ones(xs:size(1))
    tr:backward(xs, go)

    tester:eq(torch.Tensor{0, 0, 0, 0, 3, 75}, tr.gradForInput, precision)    
    tester:eq(torch.Tensor{0, 0, 0, 0, 6, 30}, tr.grad2ForInput, precision)    

    tr:zeroGradParameters()

    tester:eq(torch.Tensor(6):zero(), tr.gradForInput)
    tester:eq(torch.Tensor(6):zero(), tr.grad2ForInput)
end

return tests