local tests = torch.TestSuite()

local precision = 1e-3

function tests.ErrorCriterionSquare_testError_expected_10()
    local criterion = nnst.ErrorCriterionSquare()
    local expected = torch.Tensor{10}
    
    tester:eq(4, criterion:forward(torch.Tensor{8}, expected), precision)
    tester:eq(0, criterion:forward(torch.Tensor{10}, expected), precision)    
    tester:eq(9, criterion:forward(torch.Tensor{13}, expected), precision)    
end

function tests.ErrorCriterionSquare_testGradient_expected_10()
    local criterion = nnst.ErrorCriterionSquare()
    local expected = torch.Tensor{10}
    
    tester:eq(torch.Tensor{-4}, criterion:backward(torch.Tensor{8}, expected), precision)
    tester:eq(torch.Tensor{0}, criterion:backward(torch.Tensor{10}, expected), precision)    
    tester:eq(torch.Tensor{6}, criterion:backward(torch.Tensor{13}, expected), precision)    
end

function tests.ErrorCriterionSquare_testGradient2_expected_10()
    local criterion = nnst.ErrorCriterionSquare()
    local expected = torch.Tensor{10}

    df, ddf = criterion:backward2(torch.Tensor{8}, expected)
    tester:eq(torch.Tensor{-4}, df, precision)
    tester:eq(torch.Tensor{2}, ddf, precision)

    df, ddf = criterion:backward2(torch.Tensor{10}, expected)
    tester:eq(torch.Tensor{0}, df, precision)
    tester:eq(torch.Tensor{2}, ddf, precision)

    df, ddf = criterion:backward2(torch.Tensor{13}, expected)
    tester:eq(torch.Tensor{6}, df, precision)
    tester:eq(torch.Tensor{2}, ddf, precision)
end

function tests.ErrorCriterionSquare_testInputNotScalar()

    local function test()
        local criterion = nnst.ErrorCriterionSquare()
        criterion:forward(torch.Tensor{1, 2}, torch.Tensor{1})
    end    
    
    tester:assertError(test)
end

function tests.ErrorCriterionSquare_testTargetNotScalar()

    local function test()
        local criterion = nnst.ErrorCriterionSquare()
        criterion:forward(torch.Tensor{1}, torch.Tensor{1, 2})
    end    
    
    tester:assertError(test)
end

return tests