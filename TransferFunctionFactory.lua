local TransferFunctionFactory = {}

-- for other options see https://github.com/torch/nn/blob/master/doc/transfer.md

function TransferFunctionFactory:SoftSign()
    local result = {}

    function result:create()
        return nn.SoftSign()
    end

    function result:getName()
        return "SoftSign"
    end

    return result
end

function TransferFunctionFactory:ReLU()
    local result = {}

    function result:create()
        return nn.ReLU()
    end

    function result:getName()
        return "ReLU"
    end

    return result
end

function TransferFunctionFactory:ELU()
    local result = {}

    function result:create()
        return nn.ELU(1)
    end

    function result:getName()
        return "ELU"
    end

    return result
end

function TransferFunctionFactory:ALU()
    local result = {}

    function result:create()
        return nnst.TransferFunctionALU()
    end

    function result:getName()
        return "TransferFunctionALU"
    end

    return result
end

function TransferFunctionFactory:ReCU()
    local result = {}

    function result:create()
        return nnst.TransferFunctionReCU()
    end

    function result:getName()
        return "TransferFunctionReCU"
    end

    return result
end

function TransferFunctionFactory:getByName(name)
    if name == "SoftSign" then
        return TransferFunctionFactory:SoftSign();
    elseif name == "ReLU" then
        return TransferFunctionFactory:ReLU();    
    elseif name == "ELU" then
        return TransferFunctionFactory:ELU();
    elseif name == "TransferFunctionALU" then
        return TransferFunctionFactory:ALU();
    elseif name == "TransferFunctionReCU" then
        return TransferFunctionFactory:ReCU();
    else
        error("Unknown TransferFunctionFactory: " .. name);
    end
end

nnst.TransferFunctionFactory = TransferFunctionFactory