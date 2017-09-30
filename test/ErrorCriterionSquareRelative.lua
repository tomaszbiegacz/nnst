local tests = torch.TestSuite()

local precision = 1e-3

function tests.ErrorCriterionSquareRelative_testError_expected_10()
    local criterion = nnst.ErrorCriterionSquareRelative()
    local expected = torch.Tensor{10}
    
    tester:eq(0.04, criterion:forward(torch.Tensor{8}, expected), precision)
    tester:eq(0, criterion:forward(torch.Tensor{10}, expected), precision)    
    tester:eq(0.04, criterion:forward(torch.Tensor{12}, expected), precision)    
end

function tests.ErrorCriterionSquareRelative_testGradient_expected_10()
    local criterion = nnst.ErrorCriterionSquareRelative()
    local expected = torch.Tensor{10}
    
    tester:eq(torch.Tensor{-0.04}, criterion:backward(torch.Tensor{8}, expected), precision)
    tester:eq(torch.Tensor{0}, criterion:backward(torch.Tensor{10}, expected), precision)    
    tester:eq(torch.Tensor{0.04}, criterion:backward(torch.Tensor{12}, expected), precision)    
end

function tests.ErrorCriterionSquareRelative_testGradient2_expected_10()
    local criterion = nnst.ErrorCriterionSquareRelative()
    local expected = torch.Tensor{10}

    df, ddf = criterion:backward2(torch.Tensor{8}, expected)
    tester:eq(torch.Tensor{-0.04}, df, precision)
    tester:eq(torch.Tensor{0.02}, ddf, precision)

    df, ddf = criterion:backward2(torch.Tensor{10}, expected)
    tester:eq(torch.Tensor{0}, df, precision)
    tester:eq(torch.Tensor{0.02}, ddf, precision)

    df, ddf = criterion:backward2(torch.Tensor{12}, expected)
    tester:eq(torch.Tensor{0.04}, df, precision)
    tester:eq(torch.Tensor{0.02}, ddf, precision)
end

function tests.ErrorCriterionSquareRelative_testInputNotScalar()

    local function test()
        local criterion = nnst.ErrorCriterionSquareRelative()
        criterion:forward(torch.Tensor{1, 2}, torch.Tensor{1})
    end    
    
    tester:assertError(test)
end

function tests.ErrorCriterionSquareRelative_testTargetNotScalar()

    local function test()
        local criterion = nnst.ErrorCriterionSquareRelative()
        criterion:forward(torch.Tensor{1}, torch.Tensor{1, 2})
    end    
    
    tester:assertError(test)
end

return tests