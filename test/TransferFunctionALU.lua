local tests = torch.TestSuite()

local precision = 1e-3

function tests.TransferFunctionALU_testOutput()
    local tr = nnst.TransferFunctionALU()

    local xs = torch.Tensor{-5, -1, -0.1, 0, 1, 5}
    local result = tr:forward(xs)
    tester:eq(torch.Tensor{-1.373, -0.785, -0.1, 0, 1, 5}, result, precision)
end

local function grad2ForOutput(ddE_dda, da_db, dE_da, dda_ddb)
    return ddE_dda*da_db*da_db + dE_da*dda_ddb
end

function tests.TransferFunctionALU_testGradient()
    local tr = nnst.TransferFunctionALU()

    local xs = torch.Tensor{-5, -1, -0.1, 0, 1, 5}
    tr:forward(xs)

    local da  = torch.Tensor{1.1, 1.2, 1.3, 1.4, 1.5, 1.6}
    local dda = torch.Tensor{2.1, 2.2, 2.3, 2.4, 2.5, 2.6}
    local db, ddb = tr:backward2(xs, da, dda)

    tester:eq(torch.Tensor{0.0385, 0.5, 0.99, 1, 1, 1}, tr.gradForInput, precision)
    tester:eq(torch.Tensor{0.0385*1.1, 0.5*1.2, 0.99*1.3, 1.4, 1.5, 1.6}, tr.gradInput, precision)
    tester:eq(tr.gradInput, db, precision)
    
    tester:eq(torch.Tensor{0.0148, 0.5, 0.1961, 0, 0, 0}, tr.grad2ForInput, precision)

    tester:eq(torch.Tensor{
        grad2ForOutput(2.1, 0.0385, 1.1, 0.0148),
        grad2ForOutput(2.2, 0.5, 1.2, 0.5),
        grad2ForOutput(2.3, 0.99, 1.3, 0.1961),
        grad2ForOutput(2.4, 1, 1.4, 0),
        grad2ForOutput(2.5, 1, 1.5, 0),
        grad2ForOutput(2.6, 1, 1.6, 0)
    }, tr.grad2Input, precision)
    tester:eq(tr.grad2Input, ddb, precision)

    tr:zeroGradParameters()

    tester:eq(torch.Tensor(6):zero(), tr.gradInput)
    tester:eq(torch.Tensor(6):zero(), tr.gradForInput)

    tester:eq(torch.Tensor(6):zero(), tr.grad2ForInput)
    tester:eq(torch.Tensor(6):zero(), tr.grad2Input)
end

return tests