local ApproximationNormalizatorFactory = {}

function ApproximationNormalizatorFactory:none()
    local result = {}

    function result:normalize(value)
        return value
    end

    function result:getName()
        return "none"
    end

    return result
end

function ApproximationNormalizatorFactory:multiplicativeInverse()
    local result = {}

    function result:normalize(value)
        if value then                        
            if value >= -1 and value <= 1 then
                error("Invalid value for normalize: " .. value)
            end

            return 1 / value
        else
            error("Empty value")
        end        
    end

    function result:getName()
        return "multiplicativeInverse"
    end

    return result
end

function ApproximationNormalizatorFactory:getByName(name)
    if name == "none" then
        return ApproximationNormalizatorFactory:none();
    elseif name == "multiplicativeInverse" then
        return ApproximationNormalizatorFactory:multiplicativeInverse();
    else
        error("Unknown ApproximationNormalizatorFactory: " .. name);
    end
end

nnst.ApproximationNormalizatorFactory = ApproximationNormalizatorFactory
