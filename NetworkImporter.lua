local json = require ("dkjson")
local NetworkImporter = {}

local function importScalarFunctionNetwork(state)
    local n = nnst.NetworkScalarFunction(state)
        
    local inputLayerValues = state.layers[1]    
    for perceptonPos=1, n.config.perceptonsInLayerCount do                
        local inputLayer = n:getInputLayer()
        inputLayer:setBias(perceptonPos, inputLayerValues.bias[perceptonPos])
        inputLayer:setWeight(perceptonPos, 1, inputLayerValues.weight[perceptonPos][1])
    end
    
    if n.config.hiddenLayersCount > 0 then
        for layerPos=1, n.config.hiddenLayersCount do
            local hiddenLayerValues = state.layers[layerPos + 1]
            for perceptonPos=1, n.config.perceptonsInLayerCount do
                local hiddenLayer = n:getHiddenLayer(layerPos)
                hiddenLayer:setBias(perceptonPos, hiddenLayerValues.bias[perceptonPos])
                local weightValues = hiddenLayerValues.weight[perceptonPos]
                for inputPos=1, n.config.perceptonsInLayerCount do
                    hiddenLayer:setWeight(perceptonPos, inputPos, weightValues[inputPos])
                end
            end            
        end        
    end

    if not n:getIsPercepton() then
        for inputPos=1, n.config.perceptonsInLayerCount do
            local outputLayerValues = state.layers[n.config.hiddenLayersCount + 2]
            local weightValues = outputLayerValues.weight[1]
            n:getOutputLayer():setWeight(1, inputPos, weightValues[inputPos])
        end
    end

    return n
end

function NetworkImporter:importFromJson(jsonContent)    
    local state, pos, err = json.decode(jsonContent)
    if err then
        error("Error when reading '" .. path .. "': " .. tostring(err))
    else        
        if state.formatName == "nnst.NetworkScalarFunction" and state.formatVersion == 1 then
            return importScalarFunctionNetwork(state);
        end

        error("Unknown format for: " .. path);
    end
end

function NetworkImporter:importFromJsonFile(path)
    fd = io.open(path, 'r')
    local jsonContent = fd:read("*a")
    fd:close()

    return NetworkImporter:importFromJson(jsonContent)
end

nnst.NetworkImporter = NetworkImporter