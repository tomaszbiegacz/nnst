local tests = torch.TestSuite()

local precision = 1e-3

function tests.Linear_testOutput()
    local tr = nnst.Linear(1, 1, true)

    tr.weight[1] = 2
    tr.bias[1] = 3

    local xs = torch.Tensor{-5}
    local result = tr:forward(xs)
    tester:eq(torch.Tensor{-7}, result, precision)
end

function tests.Linear_testGradient()
    local tr = nnst.Linear(1, 1, true)

    tr.weight[1] = 2
    tr.bias[1] = 3

    local xs = torch.Tensor{-5}
    local result = tr:forward(xs)

    local go = torch.ones(1)
    tr:backward(xs, go)

    tester:eq(torch.Tensor(1, 1):fill(2), tr.gradForInput, precision)
    tester:eq(torch.Tensor(1, 1):fill(0), tr.grad2ForInput, precision)

    tr:zeroGradParameters()

    tester:eq(torch.Tensor(1, 1):fill(0), tr.gradForInput, precision)
    tester:eq(torch.Tensor(1, 1):fill(0), tr.grad2ForInput, precision)
end

function tests.Linear_testGradient_twoInputs_toOutputs()
    local tr = nnst.Linear(2, 2, true)

    -- first percepton
    tr.weight[{1, 1}] = 1.1
    tr.weight[{1, 2}] = 1.2
    tr.bias[1] = 4.1

    -- second percepton
    tr.weight[{2, 1}] = 2.1
    tr.weight[{2, 2}] = 2.2    
    tr.bias[2] = 4.2

    local xs = torch.Tensor{5, 6}
    local result = tr:forward(xs)    
    tester:eq(torch.Tensor{5*1.1+6*1.2+4.1, 5*2.1+6*2.2+4.2}, result, precision)    

    local db  = torch.Tensor{8.1, 8.2}
    local ddb = torch.Tensor{9.1, 9.2}
    local da, dda = tr:backward2(xs, db, ddb)

    local da_expected = torch.Tensor(2, 2)
    da_expected[{1, 1}] = 1.1
    da_expected[{1, 2}] = 1.2
    da_expected[{2, 1}] = 2.1
    da_expected[{2, 2}] = 2.2
    tester:eq(da_expected, tr.gradForInput, precision)
    tester:eq(torch.Tensor{8.1*1.1+8.2*2.1, 8.1*1.2+8.2*2.2}, tr.gradInput, precision)
    tester:eq(tr.gradInput, da, precision)

    tester:eq(torch.Tensor(2, 2):fill(0), tr.grad2ForInput, precision)
    local dda_expected = torch.Tensor{9.1*1.1^2 + 9.2*2.1^2, 9.1*1.2^2 + 9.2*2.2^2}    
    tester:eq(dda_expected, tr.grad2Input, precision)
    tester:eq(tr.grad2Input, dda, precision)

    local dw_expected = torch.Tensor(2, 2)
    dw_expected[{1, 1}] = db[1]*xs[1]
    dw_expected[{1, 2}] = db[1]*xs[2]
    dw_expected[{2, 1}] = db[2]*xs[1]
    dw_expected[{2, 2}] = db[2]*xs[2]    
    tester:eq(dw_expected, tr.gradWeight, precision)

    local db_expected = db    
    tester:eq(db_expected, tr.gradBias, precision)

    local ddw_expected = torch.Tensor(2, 2)
    ddw_expected[{1, 1}] = ddb[1]*xs[1]^2
    ddw_expected[{1, 2}] = ddb[1]*xs[2]^2
    ddw_expected[{2, 1}] = ddb[2]*xs[1]^2
    ddw_expected[{2, 2}] = ddb[2]*xs[2]^2
    tester:eq(ddw_expected, tr.grad2Weight, precision)

    local ddb_expected = ddb    
    tester:eq(ddb_expected, tr.grad2Bias, precision)

    tr:zeroGradParameters()

    tester:eq(torch.Tensor(2, 2):fill(0), tr.gradForInput)
    tester:eq(torch.Tensor(2):fill(0), tr.gradInput)

    tester:eq(torch.Tensor(2, 2):fill(0), tr.grad2ForInput)
    tester:eq(torch.Tensor(2):fill(0), tr.grad2Input)
end

return tests