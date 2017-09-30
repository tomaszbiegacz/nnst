local factory = nnst.ApproximationNormalizatorFactory
local tests = torch.TestSuite()

function tests.ApproximationNormalizatorFactory_NoneNormalize()
    local n = factory:none()

    tester:eq(n:normalize(2), 2)
end

function tests.ApproximationNormalizatorFactory_MultiplicativeInverseNormalize()
    local n = factory:multiplicativeInverse()

    tester:eq(n:normalize(2), 0.5)
end

function tests.ApproximationNormalizatorFactory_MultiplicativeInverseNormalizeMinus()
    local n = factory:multiplicativeInverse()

    tester:eq(n:normalize(-2), -0.5)
end

function tests.ApproximationNormalizatorFactory_MultiplicativeInverseNormalizeLowerValueError()

    local function test()
        local n = factory:multiplicativeInverse()
        n:normalize(-1)
    end    
    
    tester:assertError(test)
end

function tests.ApproximationNormalizatorFactory_MultiplicativeInverseNormalizeUpperValueError()

    local function test()
        local n = factory:multiplicativeInverse()
        n.normalize(1)
    end    
    
    tester:assertError(test)
end

function tests.ApproximationNormalizatorFactory_MultiplicativeInverseNormalizeZeroValueError()

    local function test()
        local n = factory:multiplicativeInverse()
        n:normalize(0)
    end    
    
    tester:assertError(test)
end

return tests