local tests = torch.TestSuite()

function tests.TransferFunctionFactory_factorySoftSign()
    local name = "SoftSign"
    local factory = nnst.TransferFunctionFactory:getByName(name)
    tester:eq(factory:getName(), name)
end

function tests.TransferFunctionFactory_factoryReLU()
    local name = "ReLU"
    local factory = nnst.TransferFunctionFactory:getByName(name)
    tester:eq(factory:getName(), name)
end

function tests.TransferFunctionFactory_factoryELU()
    local name = "ELU"
    local factory = nnst.TransferFunctionFactory:getByName(name)
    tester:eq(factory:getName(), name)
end

function tests.TransferFunctionFactory_factoryALU()
    local name = "TransferFunctionALU"
    local factory = nnst.TransferFunctionFactory:getByName(name)
    tester:eq(factory:getName(), name)
end

return tests