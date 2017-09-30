local json = require ("dkjson")
local NetworkScalarFunction, parent = torch.class('nnst.NetworkScalarFunction', 'nnst.NetworkSequential')

local function createPerception(container, config)                
    container:_addPerceptonLayer(
        nnst.Linear(1, 1), 
        config.transferFunctionFactory.create())
end

local function createNetworkWithHiddenLayers(container, config)        
    container:_addPerceptonLayer(
        nnst.Linear(1, config.perceptonsInLayerCount), 
        config.transferFunctionFactory.create())
    
    for i=1,config.hiddenLayersCount do
        container:_addPerceptonLayer(
            nnst.Linear(config.perceptonsInLayerCount, config.perceptonsInLayerCount), 
            config.transferFunctionFactory.create())
    end
    
    -- we don't need bias in the last layer
    container:_addPerceptonLayer(nnst.Linear(config.perceptonsInLayerCount, 1, false))
end

--
-- NetworkScalarFunction
--

function NetworkScalarFunction:__init(args)
    parent.__init(self)    

    self.config = {
        perceptonsInLayerCount = args and args.perceptonsInLayerCount or 1,
        hiddenLayersCount = args and args.hiddenLayersCount or 0,        
        transferFunctionFactory = nnst.TransferFunctionFactory:getByName(args and args.transferFunction or "TransferFunctionALU"),
        approximationNormalizer = nnst.ApproximationNormalizatorFactory:getByName(args and args.approximationNormalizer or "none")
    }

    assert(self.config.perceptonsInLayerCount >= 1, "self.config.perceptonsInLayerCount >= 1")
    assert(self.config.hiddenLayersCount >= 0, "self.config.hiddenLayersCount >= 0")

    if self:getIsPercepton() then
        createPerception(self, self.config)
    else
        createNetworkWithHiddenLayers(self, self.config)
    end
end

function NetworkScalarFunction:getIsPercepton()
    return self.config.hiddenLayersCount == 0 and self.config.perceptonsInLayerCount == 1
end

function NetworkScalarFunction:forwardScalar(value)    
    local normalizedValueTensor = torch.Tensor{ self.config.approximationNormalizer:normalize(value) }
    local outputTensor = self:forward(normalizedValueTensor) 
    return outputTensor[1]
end

function NetworkScalarFunction:forwardScalars(values)
    assert(torch.isTensor(values), "torch.isTensor(values)")
    assert(values:nDimension() == 1, "values:nDimensions() == 1")

    local count = values:size(1)
    result = torch.Tensor(count)
    for i=1,count do 
        local value = values[i]
        local normalizedValueTensor = torch.Tensor{ self.config.approximationNormalizer:normalize(value) }
        local outputTensor = self:forward(normalizedValueTensor) 
        result[i] = outputTensor[1]        
    end
    return result
end

function NetworkScalarFunction:exportToJson(path)
    local state = {
        formatName = "nnst.NetworkScalarFunction",
        formatVersion = 1,
        perceptonsInLayerCount = self.config.perceptonsInLayerCount,
        hiddenLayersCount = self.config.hiddenLayersCount,
        transferFunction = self.config.transferFunctionFactory:getName(),
        approximationNormalizer = self.config.approximationNormalizer:getName(),
        layers = self:getState()
    }

    local jsonState = json.encode(state, { indent = true })

    fd = io.open(path, 'w')
    fd:write(jsonState)
    fd:close()
end
