local tests = torch.TestSuite()

local precision = 1e-3

function tests.ErrorCriterionRelative_testDefaults()
    local criterion = nnst.ErrorCriterionRelative()

    tester:eq(1, criterion.alpha)
    tester:eq(0.2, criterion.beta)
end

function tests.ErrorCriterionRelative_testDefaultsBeta()
    local criterion = nnst.ErrorCriterionRelative(0.8)

    tester:eq(0.8, criterion.alpha)
    tester:eq(0.16, criterion.beta)
end

function tests.ErrorCriterionRelative_testError_expected_10()
    local criterion = nnst.ErrorCriterionRelative(0.8, 0.16)
    local expected = torch.Tensor{10}
    
    tester:eq(0.800, criterion:forward(torch.Tensor{2}, expected), precision)
    tester:eq(0.497, criterion:forward(torch.Tensor{5.2}, expected), precision)    
    tester:eq(0.302, criterion:forward(torch.Tensor{8.4}, expected), precision)

    tester:eq(0.275, criterion:forward(torch.Tensor{10}, expected), precision)

    tester:eq(0.302, criterion:forward(torch.Tensor{11.6}, expected), precision)    
    tester:eq(0.497, criterion:forward(torch.Tensor{14.8}, expected), precision)              
    tester:eq(0.800, criterion:forward(torch.Tensor{18}, expected), precision)
end

function tests.ErrorCriterionRelative_testGradient_expected_10()
    local criterion = nnst.ErrorCriterionRelative(0.8, 0.16)
    local expected = torch.Tensor{10}

    tester:eq(torch.Tensor{-0.1}, criterion:backward(torch.Tensor{2}, expected), precision)
    tester:eq(torch.Tensor{-0.083}, criterion:backward(torch.Tensor{5.2}, expected), precision)    
    tester:eq(torch.Tensor{-0.033}, criterion:backward(torch.Tensor{8.4}, expected), precision)    

    tester:eq(torch.Tensor{0}, criterion:backward(torch.Tensor{10}, expected), precision)

    tester:eq(torch.Tensor{0.033}, criterion:backward(torch.Tensor{11.6}, expected), precision)    
    tester:eq(torch.Tensor{0.083}, criterion:backward(torch.Tensor{14.8}, expected), precision)              
    tester:eq(torch.Tensor{0.1}, criterion:backward(torch.Tensor{18}, expected), precision)    
end

function tests.ErrorCriterionRelative_testGradient2_expected_10()
    local criterion = nnst.ErrorCriterionRelative(0.8, 0.16)
    local expected = torch.Tensor{10}

    df, ddf = criterion:backward2(torch.Tensor{2}, expected)
    tester:eq(torch.Tensor{-0.1}, df, precision)
    tester:eq(torch.Tensor{0}, ddf, precision)

    df, ddf = criterion:backward2(torch.Tensor{5.2}, expected)
    tester:eq(torch.Tensor{-0.083}, df, precision)
    tester:eq(torch.Tensor{0.010}, ddf, precision)

    df, ddf = criterion:backward2(torch.Tensor{8.4}, expected)
    tester:eq(torch.Tensor{-0.033}, df, precision)
    tester:eq(torch.Tensor{0.020}, ddf, precision)    

    df, ddf = criterion:backward2(torch.Tensor{10}, expected)
    tester:eq(torch.Tensor{0}, df, precision)
    tester:eq(torch.Tensor{0.020}, ddf, precision)    

    df, ddf = criterion:backward2(torch.Tensor{11.6}, expected)
    tester:eq(torch.Tensor{0.033}, df, precision)
    tester:eq(torch.Tensor{0.020}, ddf, precision)    

    df, ddf = criterion:backward2(torch.Tensor{14.8}, expected)
    tester:eq(torch.Tensor{0.083}, df, precision)
    tester:eq(torch.Tensor{0.010}, ddf, precision)

    df, ddf = criterion:backward2(torch.Tensor{18}, expected)
    tester:eq(torch.Tensor{0.1}, df, precision)
    tester:eq(torch.Tensor{0}, ddf, precision)
end

function tests.ErrorCriterionRelative_testError_expected_minus_10()
    local criterion = nnst.ErrorCriterionRelative(0.8, 0.16)
    local expected = torch.Tensor{-10}
    
    tester:eq(0.800, criterion:forward(torch.Tensor{-2}, expected), precision)
    tester:eq(0.497, criterion:forward(torch.Tensor{-5.2}, expected), precision)    
    tester:eq(0.302, criterion:forward(torch.Tensor{-8.4}, expected), precision)    

    tester:eq(0.275, criterion:forward(torch.Tensor{-10}, expected), precision)

    tester:eq(0.302, criterion:forward(torch.Tensor{-11.6}, expected), precision)    
    tester:eq(0.497, criterion:forward(torch.Tensor{-14.8}, expected), precision)              
    tester:eq(0.800, criterion:forward(torch.Tensor{-18}, expected), precision)    
end

function tests.ErrorCriterionRelative_testGradient_expected_minus_10()
    local criterion = nnst.ErrorCriterionRelative(0.8, 0.16)
    local expected = torch.Tensor{-10}

    tester:eq(torch.Tensor{0.1}, criterion:backward(torch.Tensor{-2}, expected), precision)
    tester:eq(torch.Tensor{0.083}, criterion:backward(torch.Tensor{-5.2}, expected), precision)    
    tester:eq(torch.Tensor{0.033}, criterion:backward(torch.Tensor{-8.4}, expected), precision)    

    tester:eq(torch.Tensor{0}, criterion:backward(torch.Tensor{-10}, expected), precision)

    tester:eq(torch.Tensor{-0.033}, criterion:backward(torch.Tensor{-11.6}, expected), precision)    
    tester:eq(torch.Tensor{-0.083}, criterion:backward(torch.Tensor{-14.8}, expected), precision)              
    tester:eq(torch.Tensor{-0.1}, criterion:backward(torch.Tensor{-18}, expected), precision)    
end

function tests.ErrorCriterionRelative_testGradient2_expected_minus_10()
    local criterion = nnst.ErrorCriterionRelative(0.8, 0.16)
    local expected = torch.Tensor{-10}

    df, ddf = criterion:backward2(torch.Tensor{-2}, expected)
    tester:eq(torch.Tensor{0.1}, df, precision)
    tester:eq(torch.Tensor{0}, ddf, precision)

    df, ddf = criterion:backward2(torch.Tensor{-5.2}, expected)
    tester:eq(torch.Tensor{0.083}, df, precision)
    tester:eq(torch.Tensor{0.010}, ddf, precision)

    df, ddf = criterion:backward2(torch.Tensor{-8.4}, expected)
    tester:eq(torch.Tensor{0.033}, df, precision)
    tester:eq(torch.Tensor{0.020}, ddf, precision)    

    df, ddf = criterion:backward2(torch.Tensor{-10}, expected)
    tester:eq(torch.Tensor{0}, df, precision)
    tester:eq(torch.Tensor{0.020}, ddf, precision)    

    df, ddf = criterion:backward2(torch.Tensor{-11.6}, expected)
    tester:eq(torch.Tensor{-0.033}, df, precision)
    tester:eq(torch.Tensor{0.020}, ddf, precision)    

    df, ddf = criterion:backward2(torch.Tensor{-14.8}, expected)
    tester:eq(torch.Tensor{-0.083}, df, precision)
    tester:eq(torch.Tensor{0.010}, ddf, precision)

    df, ddf = criterion:backward2(torch.Tensor{-18}, expected)
    tester:eq(torch.Tensor{-0.1}, df, precision)
    tester:eq(torch.Tensor{0}, ddf, precision)
end

function tests.ErrorCriterionRelative_testInputNotScalar()

    local function test()
        local criterion = nnst.ErrorCriterionRelative()
        criterion:forward(torch.Tensor{1, 2}, torch.Tensor{1})
    end    
    
    tester:assertError(test)
end

function tests.ErrorCriterionRelative_testTargetNotScalar()

    local function test()
        local criterion = nnst.ErrorCriterionRelative()
        criterion:forward(torch.Tensor{1}, torch.Tensor{1, 2})
    end    
    
    tester:assertError(test)
end

function tests.ErrorCriterionRelative_testExpectedZero()

    local function test()
        local criterion = nnst.ErrorCriterionRelative()
        criterion:forward(torch.Tensor{1}, torch.Tensor{0})
    end    
    
    tester:assertError(test)
end

return tests